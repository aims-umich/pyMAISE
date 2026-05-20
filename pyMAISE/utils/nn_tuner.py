import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import optuna
import torch
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.utils.multiclass import type_of_target

import pyMAISE.settings as settings
from .trial import determine_class_from_probabilities


def _run_fold_subprocess(
    hypermodel,
    trial_params,
    xtrain,
    xval,
    ytrain,
    yval,
    y_all,
    device,
    metrics,
    problem_type,
):
    """
    Run one CV fold in a subprocess.

    This is a module-level function (not a method) so that
    ``concurrent.futures.ProcessPoolExecutor`` can pickle it for dispatch to
    worker processes.

    The live Optuna ``trial`` object is not picklable, so ``trial_params``
    (a plain dict) is passed and replayed via ``FixedTrial``.  Each subprocess
    builds a fresh model with the same hyperparameter values but independently
    initialized weights, then assigns it to the requested ``device`` before
    fitting.
    """
    import optuna as _optuna
    from pyMAISE.settings import ProblemType
    from pyMAISE.utils.trial import determine_class_from_probabilities as _dcfp

    fixed_trial = _optuna.trial.FixedTrial(trial_params)

    # Build the model then assign the GPU before fitting.
    # skorch reads self.device in NeuralNet.initialize(), which runs at the
    # start of fit(), so set_params() must be called beforehand.
    model = hypermodel.build(fixed_trial)
    model.set_params(device=device)

    hypermodel.fit(fixed_trial, model, xtrain, ytrain)

    yval_pred = model.predict(xval)

    if problem_type == ProblemType.CLASSIFICATION:
        yval_pred = _dcfp(yval_pred, y_all)

    if metrics is not None:
        return float(
            metrics(
                yval_pred.reshape(-1, yval.shape[-1]),
                yval.reshape(-1, yval.shape[-1]),
            )
        )

    if problem_type == ProblemType.CLASSIFICATION:
        return 1.0 - float(np.mean(yval_pred.reshape(-1) == yval.reshape(-1)))
    return float(np.mean((yval_pred - yval) ** 2))


class NNTuner:
    """
    Hyperparameter tuner for pyMAISE neural networks using Optuna.

    Replaces the keras-tuner ``Tuner`` base class.  Each call to ``search()``
    creates a fresh Optuna study, runs ``n_trials`` trials, and evaluates each
    trial via k-fold cross-validation.

    When ``settings.values.run_parallel`` is ``True`` and at least two CUDA
    GPUs are available, the CV folds within each trial are distributed across
    GPUs using ``ProcessPoolExecutor``.  If fewer than two GPUs are found a
    ``UserWarning`` is issued and execution falls back to serial.  Trials
    themselves always run sequentially so that all Optuna samplers (including
    ``GridSampler`` and ``TPESampler``) work without a shared storage backend.

    Parameters
    ----------
    hypermodel: nnHyperModel
        Hypermodel whose ``build()`` and ``fit()`` methods are called per fold.
    sampler: optuna.samplers.BaseSampler
        Optuna sampler (e.g. ``GridSampler``, ``TPESampler``, ``RandomSampler``).
    n_trials: int
        Number of Optuna trials to run.  For grid search this should equal the
        total number of grid points so the sampler exhausts the space exactly once.
    objective: str
        Display name shown in progress output.  Not used by Optuna internally.
    cv: int or sklearn CV splitter
        Number of folds or a pre-configured splitter with a ``split(x, y)`` method.
    shuffle: bool
        Whether to shuffle before splitting (only used when ``cv`` is an int).
    metrics: callable or None
        ``metrics(y_pred, y_true) -> float`` scoring function.  When ``None`` a
        default is used: MSE for regression, error rate for classification.
    direction: str
        ``"minimize"`` or ``"maximize"``; passed directly to ``optuna.create_study``.
    """

    def __init__(
        self,
        hypermodel,
        sampler,
        n_trials,
        objective="score",
        cv=5,
        shuffle=True,
        metrics=None,
        direction="minimize",
    ):
        self.hypermodel = hypermodel
        self._sampler = sampler
        self._n_trials = n_trials
        self._objective = objective
        self._cv = cv
        self._shuffle = shuffle
        self._metrics = metrics
        self._direction = direction

        self._study = None
        # Stores per-fold scores for each trial so std_test_score can be computed.
        self._trial_fold_scores = {}

    # =======================================================================
    # Methods

    def search(self, x, y):
        """
        Run the hyperparameter search.

        Parameters
        ----------
        x: numpy.ndarray
            Input features.
        y: numpy.ndarray
            Target values.
        """
        # Initialize CV splitter (turns an int into KFold or StratifiedKFold)
        self._cv = self._init_cv(self._cv, self._shuffle, y)

        # In quiet mode, suppress per-trial Optuna output and show a progress
        # bar instead.  In verbose mode, show Optuna trial details.
        if settings.values.verbosity == 0:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            print(f"Tuning {self.hypermodel._name}")
        else:
            optuna.logging.set_verbosity(optuna.logging.INFO)

        self._study = optuna.create_study(
            direction=self._direction,
            sampler=self._sampler,
        )

        self._study.optimize(
            lambda trial: self._run_trial(trial, x, y),
            n_trials=self._n_trials,
            show_progress_bar=settings.values.verbosity == 0,
        )

    def _run_trial(self, trial, x, y):
        """
        Objective function passed to ``study.optimize()``.

        Determines whether to run folds in parallel or serial, then collects
        and records scores for this trial.
        """
        fold_splits = list(self._cv.split(x, y))

        use_parallel = settings.values.run_parallel
        if use_parallel:
            n_gpus = torch.cuda.device_count()
            if n_gpus < 2:
                # Gracefully fall back to serial when the hardware can't support
                # fold-level parallelism (0 GPUs = no acceleration at all; 1 GPU
                # = folds would compete for memory on the same device).
                # Python's warnings module deduplicates by default, so this
                # message appears at most once per session.
                warnings.warn(
                    f"run_parallel=True requested but only {n_gpus} GPU(s) detected. "
                    "Falling back to serial execution.",
                    UserWarning,
                    stacklevel=2,
                )
                use_parallel = False

        if use_parallel:
            scores = self._run_folds_parallel(trial, x, y, fold_splits, n_gpus)
        else:
            scores = self._run_folds_serial(trial, x, y, fold_splits)

        self._trial_fold_scores[trial.number] = scores
        return float(np.mean(scores))

    def _run_folds_serial(self, trial, x, y, fold_splits):
        """Run each CV fold sequentially in the current process."""
        scores = []
        for train_idx, val_idx in fold_splits:
            xtrain, xval = x[train_idx], x[val_idx]
            ytrain, yval = y[train_idx], y[val_idx]

            # build() is safe to call multiple times per trial: Optuna caches
            # suggest_* results within a trial, so the same architecture is
            # reproduced each fold while weights are re-initialized from scratch.
            model = self.hypermodel.build(trial)
            self.hypermodel.fit(trial, model, xtrain, ytrain)

            scores.append(self._evaluate(model, xval, yval, y))
        return scores

    def _run_folds_parallel(self, trial, x, y, fold_splits, n_gpus):
        """
        Dispatch each CV fold to a subprocess pinned to a distinct GPU.

        Folds are assigned to GPUs round-robin (``fold_idx % n_gpus``), so
        with 5 folds and 2 GPUs: folds 0, 2, 4 → cuda:0 and folds 1, 3 →
        cuda:1.  ``ProcessPoolExecutor`` runs at most ``n_gpus`` subprocesses
        simultaneously, naturally load-balancing across devices.
        """
        problem_type = settings.values.problem_type

        # Populate trial.params by sampling once on the main process.
        # FixedTrial in each subprocess requires all parameter names to be
        # present at construction time, but trial.params is empty until
        # build() calls suggest_* at least once.  The built model is discarded.
        self.hypermodel.build(trial)
        trial_params = trial.params

        futures = {}
        with ProcessPoolExecutor(max_workers=n_gpus) as executor:
            for fold_idx, (train_idx, val_idx) in enumerate(fold_splits):
                fut = executor.submit(
                    _run_fold_subprocess,
                    self.hypermodel,
                    trial_params,
                    x[train_idx],
                    x[val_idx],
                    y[train_idx],
                    y[val_idx],
                    y,
                    f"cuda:{fold_idx % n_gpus}",
                    self._metrics,
                    problem_type,
                )
                futures[fut] = fold_idx

            # Collect results preserving fold order
            scores_by_fold = {}
            for fut in as_completed(futures):
                scores_by_fold[futures[fut]] = fut.result()

        return [scores_by_fold[i] for i in range(len(fold_splits))]

    def _evaluate(self, model, xval, yval, y_all):
        """Score a fitted skorch model on one validation fold (serial path)."""
        # skorch predict() accepts numpy arrays and returns numpy arrays
        yval_pred = model.predict(xval)

        if settings.values.problem_type == settings.ProblemType.CLASSIFICATION:
            yval_pred = determine_class_from_probabilities(yval_pred, y_all)

        if self._metrics is not None:
            return float(
                self._metrics(
                    yval_pred.reshape(-1, yval.shape[-1]),
                    yval.reshape(-1, yval.shape[-1]),
                )
            )

        # Default fallback so search() works when metrics is None
        if settings.values.problem_type == settings.ProblemType.CLASSIFICATION:
            return 1.0 - float(np.mean(yval_pred.reshape(-1) == yval.reshape(-1)))
        return float(np.mean((yval_pred - yval) ** 2))

    # =======================================================================
    # Static Methods

    @staticmethod
    def _init_cv(cv, shuffle, y):
        """Turn an integer fold count into a KFold or StratifiedKFold object."""
        if isinstance(cv, int):
            if (
                settings.values.problem_type == settings.ProblemType.CLASSIFICATION
                and type_of_target(y) in ("binary", "multiclass")
            ):
                return StratifiedKFold(
                    n_splits=cv,
                    shuffle=shuffle,
                    random_state=settings.values.random_state if shuffle else None,
                )
            else:
                return KFold(
                    n_splits=cv,
                    shuffle=shuffle,
                    random_state=settings.values.random_state if shuffle else None,
                )

        return cv

    # =======================================================================
    # Getters/Setters

    @property
    def mean_test_score(self):
        """Mean CV score for each completed trial, in trial order."""
        return [t.value for t in self._study.trials]

    @property
    def std_test_score(self):
        """Standard deviation of CV fold scores for each completed trial."""
        return [
            float(np.std(self._trial_fold_scores[t.number]))
            for t in self._study.trials
        ]

    @property
    def study(self):
        """The underlying ``optuna.Study`` object."""
        return self._study
