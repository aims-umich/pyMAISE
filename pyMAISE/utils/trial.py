from multiprocessing import Manager, Process
from enum import Enum

import numpy as np
import tensorflow as tf
from sklearn.utils.multiclass import type_of_target

from .process_pool import ProcessPool
from .hyperparameters import HyperParameters
import pyMAISE.settings as settings


class TrialStatus(Enum):
    """
    Status Enum for Trial object.

    - ``INITIALIZED``: Trial object has been initialized.
    - ``RUNNING``: Trial has started and is ready for submitting jobs.
    - ``ALL_SUBMITTED``: All cross-validation splits have been submitted
      but the jobs are not done running.
    - ``FINISHED``: All cross-validation splits are done running.
    """

    INITIALIZED = 1
    RUNNING = 2
    ALL_SUBMITTED = 3
    FINISHED = 4


class Trial(object):
    """
    Trial object for each iteration of an NN search algorithm. This object handles all
    cross-validation (CV) splits for the given trial.

    Parameters
    ----------
    hypermodel: pyMAISE.methods.nn.nnHyperModel
        NN hypermodel for generating Keras NN models.
    kt_trial: keras_tuner.engine.trial.Trial
        Keras-Tuner trial.
    metrics: None or callable
        Function for computing the objective function.
    objective: str
        Objective funtion name.
    """

    def __init__(self, hypermodel, kt_trial, metrics, objective):
        # Save params
        self._hypermodel = hypermodel
        self._kt_trial = kt_trial
        self._metrics = metrics
        self._objective = objective

        self._pids = []
        self._manager = None
        self._test_scores = []
        self._status = TrialStatus.INITIALIZED

    # =======================================================================
    # Methods

    def start_trial(self, cv, inputs, outputs, process_pool: ProcessPool):
        """
        Pre-trial method setup. Includes creating CV generator, updating status, and
        initializing the Trials parallel environment.

        Parameters
        ----------
        cv: object
            CV object with ``split(x, y)`` method for splitting the ``inputs``
            and ``outputs``.
        inputs: numpy.ndarray
            Input data.
        outputs: numpy.ndarray
            Ouput data.
        process_pool: ProcessPool or None
            ``ProcessPool`` for parallel execution or ``None`` for serial
            execution.
        """
        self._process_pool = process_pool

        # Save parameters
        self._inputs = inputs
        self._outputs = outputs

        # Setup generator
        self._generator = cv.split(inputs, outputs)

        # Change trial status
        self._status = TrialStatus.RUNNING

        if process_pool:
            # Initialize manager
            self._manager = Manager()

            # Determine memory requirement for one model
            self._namespace = self._manager.Namespace()
            self._namespace.mem_estimate = 0

            # Run process
            kt_trial = self._manager.list([self._kt_trial])
            process = Process(
                target=self._compute_model_memory,
                args=(self._namespace, self._hypermodel, kt_trial),
            )
            process.start()
            process.join()
            assert process.exitcode == 0
            process.terminate()

            self._kt_trial = kt_trial[0]

            # Add 50 MB head room and get ciel
            self._namespace.mem_estimate = int(self._namespace.mem_estimate + 50) + 1

            # Ensure model is trainable on a GPU
            process_pool.check_model_size(self._namespace.mem_estimate)

            # Add list for test scores
            self._test_scores = self._manager.list()

    def clean_processes(self):
        """
        Remove all processes that are no longer alive that belong to this trial.

        Returns
        -------
        n: int
            Number of processes terminated.
        """
        # Terminate finished processes
        n = 0
        for i, pid in enumerate(self._pids):
            # Check if process is still running
            if not self._process_pool.is_alive(pid):
                # Remove process ID if it's still running
                self._pids.pop(i)
                n += 1

        # If all processes have run and there are no more
        # splits we can terminate the trial
        if not self._pids and self._status == TrialStatus.ALL_SUBMITTED:
            # Get test scores form manager
            self._test_scores = list(self._test_scores)

            # Update status
            self._status = TrialStatus.FINISHED

        return n

    def add_process_batch(self):
        """
        Add batch of processes. Iterates through available CV splits and eligible GPUs.
        Terminates when there are no more eligible GPUs or all CV splits have been
        submitted.

        Returns
        -------
        n: int
            Number of processes submitted.
        """
        n = 0
        while True:
            # Find eligible device
            device_idx = self._process_pool.find_eligible_device(
                self._namespace.mem_estimate
            )

            if device_idx is not None:
                # Get next CV split
                split = next(self._generator, None)

                if split:
                    # Send split to ProcessPool
                    self._pids.append(
                        self._process_pool.submit_process(
                            target=self._run_parallel_model,
                            args=(
                                self._namespace,
                                self._test_scores,
                                self._metrics,
                                self._objective,
                                settings.values.problem_type,
                                settings.values.verbosity,
                                self._hypermodel,
                                self._kt_trial.hyperparameters,
                                (self._inputs, self._outputs),
                                split,
                                self._process_pool.devices[device_idx].id,
                            ),
                            device_idx=device_idx,
                            job_memory=self._namespace.mem_estimate,
                        )
                    )
                    n += 1

                else:
                    self._status = TrialStatus.ALL_SUBMITTED
                    return n

            else:
                return n

    def serial_cv(self, p):
        """
        Run through all CV splits in serial.

        Parameters
        ----------
        p: tqdm or None
            tqdm progress bar.
        """
        while True:
            split = next(self._generator, None)

            if split:
                self._run_model(
                    self._test_scores,
                    self._metrics,
                    self._objective,
                    settings.values.problem_type,
                    settings.values.verbosity,
                    self._hypermodel,
                    self._kt_trial.hyperparameters,
                    (self._inputs, self._outputs),
                    split,
                )

                # Update progress bar
                if p:
                    p.n += 1
                    p.refresh()

            else:
                self._status = TrialStatus.ALL_SUBMITTED
                return

    def finalize(self):
        """
        Finalize trial. Shutdown multiprocessing.Manager if running in parallel.

        Returns
        -------
        mean_test_score: float
            Mean CV test score.
        std_test_score: float
            Standard deviation of mean CV test score.
        """
        if self._manager:
            # Shutdown manager
            self._manager.shutdown()

        # Compute statistics
        return np.mean(self._test_scores), np.std(self._test_scores)

    # =======================================================================
    # Static Methods

    @staticmethod
    def _compute_model_memory(*args):
        """
        Estimate an NN model's memory.

        Parameters
        ----------
        ns: multiprocessing.Manager.Namespace
            Namespace with ``mem_estimate`` parameter.
        hypermodel: pyMAISE.methods.nn.nnHyperModel
            NN hypermodel for building Keras models.
        kt_trial: list of one keras_tuner.engine.trial.Trial
            Keras-Tuner trial with hyperparameters.
        """
        # Extract arguments
        ns, hypermodel, kt_trial = args
        trial = kt_trial[0]

        # Get batch_size
        if "batch_size" not in hypermodel._fitting_params:
            raise RuntimeError("All neural networks must have a batch_size")

        batch_size = hypermodel._fitting_params["batch_size"]
        if isinstance(batch_size, HyperParameters):
            batch_size = batch_size.hp(trial.hyperparameters, "batch_size")

        # Estimate a models memory footprint
        mem_estimate = ns.mem_estimate

        # Create model
        model = hypermodel.build(trial.hyperparameters)
        kt_trial[0] = trial

        # Estimate activation
        for layer in model.layers:
            output_shape = layer.output.shape

            if None in output_shape:
                mem_estimate += batch_size * tf.reduce_prod(output_shape[1:]).numpy()

        # Assume optimizer takes 3x memory of parameters (rule of thumb for Adam)
        mem_estimate += 3 * model.count_params()

        # Add conservative memory constraint
        mem_estimate *= 1.5

        # Determine floating point precision
        float_type = model.dtype_policy
        mem_estimate /= 1024**2

        if float_type == "float16":
            mem_estimate *= 2.0
        elif float_type == "float64":
            mem_estimate *= 8.0
        else:
            # Assume float32
            mem_estimate *= 4.0

        ns.mem_estimate = mem_estimate

    @staticmethod
    def _run_parallel_model(*args):
        """
        Run a model in parallel.

        Parameters
        ----------
        ns: multiprocessing.Manager.Namespace
            Namespace with model memory estimate (``mem_estimate``).
        test_scores: multiprocessing.Manager.list
            List for test scores.
        metrics: None or callable
            Function for computing the test score.
        objective: str
            Name of objective function.
        problem_type: pyMAISE.ProblemType
            Problem type, regression or classification.
        verbose: int
            Level of output
        hypermodel: pyMAISE.methods.nn.nnHyperModel
            NN hypermodel for building Keras models.
        hp: list of one keras_tuner.engine.trial.Trial.HyperParameters
            Hyperparameters for building the NN model.
        training_data: tuple of numpy.ndarray
            Training data including ``(xtrain, ytrain)``
        split_idxs: tuple of numpy.ndarray
            Indices for splitting the training data into a training
            and validation set.
        device: tf.config.PhysicalDevice
            Device to train the model on.
        """
        # Extract arguments
        (
            ns,
            test_scores,
            metrics,
            objective,
            problem_type,
            verbose,
            hypermodel,
            hp,
            (x, y),
            (train_idxs, val_idxs),
            device,
        ) = args

        # Limit visible devices to given device and CPUs
        tf.config.set_visible_devices([device] + tf.config.list_physical_devices("CPU"))

        # Limit memory by making a virtual device
        tf.config.set_logical_device_configuration(
            device,
            [tf.config.LogicalDeviceConfiguration(memory_limit=ns.mem_estimate)],
        )

        # Run model configuration with virtual device
        with tf.device(tf.config.list_logical_devices("GPU")[0]):
            Trial._run_model(
                test_scores,
                metrics,
                objective,
                problem_type,
                verbose,
                hypermodel,
                hp,
                (x, y),
                (train_idxs, val_idxs),
            )

    @staticmethod
    def _run_model(*args):
        """
        Run a model.

        Parameters
        ----------
        test_scores: multiprocessing.Manager.list
            List for test scores.
        metrics: None or callable
            Function for computing the test score.
        objective: str
            Name of objective function.
        problem_type: pyMAISE.ProblemType
            Problem type, regression or classification.
        verbose: int
            Level of output
        hypermodel: pyMAISE.methods.nn.nnHyperModel
            NN hypermodel for building Keras models.
        hp: list of one keras_tuner.engine.trial.Trial.HyperParameters
            Hyperparameters for building the NN model.
        training_data: tuple of numpy.ndarray
            Training data including ``(xtrain, ytrain)``
        split_idxs: tuple of numpy.ndarray
            Indices for splitting the training data into a training
            and validation set.
        """
        # Extract arguments
        (
            test_scores,
            metrics,
            objective,
            problem_type,
            verbose,
            hypermodel,
            hp,
            (x, y),
            (train_idxs, val_idxs),
        ) = args

        # Get training/validation split
        xtrain, xval = x[train_idxs,], x[val_idxs,]
        ytrain, yval = y[train_idxs,], y[val_idxs,]

        # Build and fit hypermodel
        model = hypermodel.build(hp)
        hypermodel.fit(hp, model, xtrain, ytrain)

        # Evaluate model performance
        if metrics is not None:
            yval_pred = model.predict(xval, verbose=verbose)

            # Round probabilities to correct class based on data format
            if problem_type == settings.ProblemType.CLASSIFICATION:
                yval_pred = determine_class_from_probabilities(yval_pred, y)

            test_scores.append(
                metrics(
                    yval_pred.reshape(-1, yval.shape[-1]),
                    yval.reshape(-1, yval.shape[-1]),
                )
            )

        else:
            test_score_idx = model.metrics_names.index(objective)
            test_scores.append(model.evaluate(xval, yval)[test_score_idx])

    # =======================================================================
    # Getters/Setters

    @property
    def status(self):
        return self._status

    @property
    def kt_trial(self):
        return self._kt_trial


def determine_class_from_probabilities(y_pred, y):
    """
    Determine the whether a given classification problem is binary or multiclass.

    Parameters
    ----------
    y_pred: numpy.ndarray
        Output predicted by model. Probability for each class.
    y: numpy.ndarray
        Expected output.

    Returns
    -------
    y_pred: numpy.ndarray
        Classification based on given probability.
    """
    if type_of_target(y) == "binary" or type_of_target(y) == "multiclass":
        # Round to nearest number
        return np.round(y_pred)

    elif type_of_target(y) == "multilabel-indicator":
        assert np.all((y_pred <= 1) & (y_pred >= 0))
        # Assert 0 or 1 for one hot encoding
        return np.where(y_pred == y_pred.max(axis=-1, keepdims=True), 1, 0)
