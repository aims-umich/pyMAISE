from multiprocessing import Process, Manager
import copy
import os

import numpy as np
import keras_tuner as kt
from sklearn.model_selection import KFold, StratifiedKFold
from tqdm.auto import tqdm

import pyMAISE.settings as settings
from .display import _try_clear
from .process_pool import ProcessPool
from .trial import Trial, TrialStatus
from .tuner_utils import validate_trial_results


class NNTuner(kt.engine.tuner.Tuner):
    def __init__(
        self,
        oracle,
        hypermodel,
        objective,
        cv=5,
        shuffle=True,
        metrics=None,
        directory=None,
        project_name=None,
        tuner_id=None,
        overwrite=False,
        executions_per_trial=1,
        verbose=0,
        **kwargs,
    ):
        self.oracle = oracle
        self.hypermodel = hypermodel
        self._objective = objective
        self._cv = self._init_cv(cv, shuffle)
        self._metrics = metrics
        self._verbose = verbose
        self._executions_per_trial = executions_per_trial
        self._run_parallel = True

        self._running_trials = []
        self._p = None

        if not isinstance(oracle, kt.engine.oracle.Oracle):
            raise ValueError(
                "Expected `oracle` argument to be an instance of `Oracle`. "
                f"Received: oracle={oracle} (of type ({type(oracle)})."
            )

        if len(kwargs) > 0:
            raise ValueError(
                f"Unrecognized arguments {list(kwargs.keys())} "
                "for `BaseTuner.__init__()`."
            )

        # Ops and metadata
        self.directory = directory or "."
        self.project_name = project_name or "untitled_project"
        self.oracle._set_project_dir(self.directory, self.project_name)

        # To support tuning distribution.
        self.tuner_id = os.environ.get("KERASTUNER_TUNER_ID", "tuner0")

        # Reloading state.
        self._populate_initial_space()

        # Initialize progress bar
        if self._verbose == 0:
            num_trials = (
                self.oracle.max_trials
                if self.oracle.max_trials
                else np.prod(
                    [len(hp._values) for hp in self.hypermodel.get_hyperparameters()]
                )
            ) * self._cv.n_splits

            self._p = tqdm(range(int(num_trials)), desc=self.hypermodel._name)

        else:
            print(f"Tuning {self.hypermodel._name}")

    # =======================================================================
    # Methods

    def _populate_initial_space(self):
        # Declare_hyperparameters is not overriden.
        hp = self.oracle.get_space()
        self.hypermodel.declare_hyperparameters(hp)
        self.oracle.update_space(hp)

        with Manager() as manager:
            hps = manager.list()

            process = Process(
                target=self._activate_conditions,
                args=(hps, self.hypermodel, self.oracle),
            )
            process.start()
            process.join()
            assert process.exitcode == 0
            process.terminate()

            # Update hyperparameter space
            self.oracle.hyperparameters = hps[0]

            hp.ensure_active_values()

    def search(self, x, y, *fit_args, **fit_kwargs):
        if "verbose" in fit_kwargs:
            self._verbose = fit_kwargs.get("verbose")
            self.oracle.verbose = self._verbose

        {True: self._parallel_search, False: self._serial_search}[
            settings.values.run_parallel
        ](x, y)

    def _parallel_search(self, x, y):
        # Create process pool
        process_pool = ProcessPool()

        # Run search
        self.on_search_begin()

        submitted_last_trial = False

        while True:
            # Clean each current running trial from finished processes
            n = 0
            for trail in self._running_trials:
                n += trail.clean_processes()

            if self._p:
                self._p.n += n
                self._p.refresh()

            # Iterate through existing trials
            create_new_trial = True
            for i, trial in enumerate(self._running_trials):
                if trial.status == TrialStatus.RUNNING:
                    # Add additional splits and don't create any new trials
                    trial.add_process_batch()
                    create_new_trial = False

                elif trial.status == TrialStatus.FINISHED:
                    # End trial
                    self.end_trial(trial)

                    # Remove from trial pool
                    self._running_trials.pop(i)

                elif trial.status == TrialStatus.ALL_SUBMITTED and isinstance(
                    self.oracle, kt.oracles.BayesianOptimizationOracle
                ):
                    # Only create one trial at a time
                    create_new_trial = False

            if create_new_trial and not submitted_last_trial:
                # Create keras tuner trial
                self.pre_create_trial()
                kt_trial = self._create_trial(self.tuner_id)

                # Check if all trails are submitted
                if kt_trial.status == kt.engine.trial.TrialStatus.STOPPED:
                    submitted_last_trial = True

                else:
                    # Add trial and start it
                    self.on_trial_begin(kt_trial)
                    self._running_trials[-1].start_trial(
                        cv=self._cv,
                        inputs=x,
                        outputs=y,
                        process_pool=process_pool,
                    )

            # If all trials are done then finalize and break
            elif submitted_last_trial and not self._running_trials:
                self.on_search_end()

                if self._verbose == 0:
                    _try_clear()

                return

    def _serial_search(self, x, y):
        # Start search
        self.on_search_begin()

        while True:
            if self._p:
                self._p.refresh()

            # Create new keras tuner trial
            self.pre_create_trial()
            kt_trial = self._create_trial(self.tuner_id)

            # If there are no more trials then finish search
            if kt_trial.status == kt.engine.trial.TrialStatus.STOPPED:
                self.on_search_end()

                if self._verbose == 0:
                    _try_clear()

                return

            # Initialize trial
            self.on_trial_begin(kt_trial)
            trial = self._running_trials.pop()
            trial.start_trial(
                cv=self._cv,
                inputs=x,
                outputs=y,
                process_pool=None,
            )

            # Run through all CV splits
            trial.serial_cv(self._p)

            # Finalize trial
            self.end_trial(trial)

    def on_trial_begin(self, trial):
        # Append trial to running trails
        self._running_trials.append(
            Trial(
                hypermodel=self.hypermodel,
                kt_trial=trial,
                metrics=self._metrics,
                objective=self._objective,
            )
        )

    def end_trial(self, trial):
        # Finalize trial and save scores
        mean_test_score, std_test_score = trial.finalize()
        trial.kt_trial.status = kt.engine.trial.TrialStatus.COMPLETED

        # Add results to oracle and keras tuner trial
        result = {
            self._objective: mean_test_score,
            self._objective + "_std": std_test_score,
        }

        validate_trial_results(result, self.oracle.objective, "Tuner.run_trial()")
        self.oracle.update_trial(trial.kt_trial.trial_id, metrics=result)
        self.on_trial_end(trial.kt_trial)

    def _create_trial(self, tuner_id):
        # Make the trial_id the current number of trial, pre-padded with 0s
        trial_id = f"{{:0{len(str(self.oracle.max_trials))}d}}"
        trial_id = trial_id.format(len(self.oracle.trials))

        if self.oracle.max_trials and len(self.oracle.trials) >= self.oracle.max_trials:
            status = kt.engine.trial.TrialStatus.STOPPED
            values = None
        else:
            response = self.oracle.populate_space(trial_id)
            status = response["status"]
            values = response["values"] if "values" in response else None

        hyperparameters = self.oracle.hyperparameters.copy()
        hyperparameters.values = values or {}

        trial = kt.engine.trial.Trial(
            hyperparameters=hyperparameters, trial_id=trial_id, status=status
        )

        if status == kt.engine.trial.TrialStatus.RUNNING:
            # Record the populated values (active only). Only record when the
            # status is RUNNING. If other status, the trial will not run, the
            # values are discarded and should not be recorded, in which case,
            # the trial_id may appear again in the future.
            self.oracle._record_values(trial)

            self.oracle.ongoing_trials[tuner_id] = trial
            self.oracle.trials[trial_id] = trial
            self.oracle.start_order.append(trial_id)
            self.oracle._save_trial(trial)
            self.save()
            self.oracle._display.on_trial_begin(trial)

        return trial

    # =======================================================================
    # Static Methods
    @staticmethod
    def _init_cv(cv, shuffle):
        if isinstance(cv, int):
            if settings.values.problem_type == settings.ProblemType.CLASSIFICATION:
                return StratifiedKFold(
                    n_splits=cv,
                    shuffle=shuffle,
                    random_state=(
                        settings.values.random_state if shuffle is True else None
                    ),
                )
            else:
                return KFold(
                    n_splits=cv,
                    shuffle=shuffle,
                    random_state=(
                        settings.values.random_state if shuffle is True else None
                    ),
                )

        return cv

    @staticmethod
    def _activate_conditions(*args):
        hps, hypermodel, oracle = args

        # Lists of stacks of conditions used during `explore_space()`.
        scopes_never_active = []
        scopes_once_active = []

        hp = oracle.get_space()
        while True:
            hypermodel.build(hp)
            oracle.update_space(hp)

            # Update the recorded scopes.
            for conditions in hp.active_scopes:
                if conditions not in scopes_once_active:
                    scopes_once_active.append(copy.deepcopy(conditions))
                if conditions in scopes_never_active:
                    scopes_never_active.remove(conditions)
            for conditions in hp.inactive_scopes:
                if conditions not in scopes_once_active:
                    scopes_never_active.append(copy.deepcopy(conditions))

            # All conditional scopes are activated.
            if not scopes_never_active:
                break

            # Generate new values to activate new conditions.
            hp = oracle.get_space()
            conditions = scopes_never_active[0]
            for condition in conditions:
                hp.values[condition.name] = condition.values[0]

            hp.ensure_active_values()

        hps += [oracle.get_space()]

    # =======================================================================
    # Getters/Setters

    @property
    def mean_test_score(self):
        return [trial.score for trial in self.oracle.trials.values()]

    @property
    def std_test_score(self):
        return [
            trial.metrics.get_history(self._objective + "_std")[0].value[0]
            for trial in self.oracle.trials.values()
        ]
