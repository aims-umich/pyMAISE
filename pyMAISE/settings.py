import os
import random
import warnings
from enum import Enum

import numpy as np
import tensorflow as tf


class ProblemType(Enum):
    """Enum to define the problem type."""

    #: pyMAISE.ProblemType: Set for a regression problem.
    REGRESSION = 0

    #: pyMAISE.ProblemType: Set for a classification problem.
    CLASSIFICATION = 1

    @classmethod
    def _get_member(cls, problem_type):
        if isinstance(problem_type, str):
            for p in ProblemType:
                if problem_type.upper() == p._name_:
                    return p
        else:
            return ProblemType(problem_type)

        raise ValueError(f"{problem_type} is not a member of {ProblemType.__name__}")


# Class for global settings
class Settings:
    def __init__(self, problem_type, **kwargs):
        self._problem_type = ProblemType._get_member(problem_type)

        # Defaults
        self._verbosity = kwargs.get("verbosity", 0)
        self._random_state = kwargs.get("random_state", None)
        self._num_configs_saved = kwargs.get("num_configs_saved", 5)
        self._new_nn_architecture = kwargs.get("new_nn_architecture", True)
        self._cuda_visible_devices = kwargs.get("cuda_visible_devices", None)

        if self._cuda_visible_devices is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = self._cuda_visible_devices

        if self._verbosity <= 1:
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
            tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
            tf.keras.utils.disable_interactive_logging()
        else:
            print(
                "Num GPUs Available: ",
                len(tf.config.experimental.list_physical_devices("GPU")),
            )

            warnings.simplefilter(action="ignore", category=Warning)
            warnings.simplefilter(action="ignore", category=FutureWarning)

        if self._random_state is not None:
            os.environ["PYTHONHASHSEED"] = str(self._random_state)
            random.seed(self._random_state)
            np.random.seed(self._random_state)
            tf.compat.v1.set_random_seed(self._random_state)

            # Deterministic tensorflow
            os.environ["TF_DETERMINISTIC_OPS"] = "1"
            os.environ["TF_CUBNN_DETERMINISTIC"] = "1"

    # Getters
    @property
    def problem_type(self) -> ProblemType:
        return self._problem_type

    @property
    def verbosity(self) -> int:
        return self._verbosity

    @property
    def random_state(self) -> int:
        return self._random_state

    @property
    def num_configs_saved(self) -> int:
        return self._num_configs_saved

    @property
    def new_nn_architecture(self) -> bool:
        return self._new_nn_architecture

    @property
    def cuda_visible_devices(self):
        return self._cuda_visible_devices

    # Setters
    @problem_type.setter
    def problem_type(self, problem_type):
        self._problem_type = ProblemType.get_member(problem_type)

    @verbosity.setter
    def verbosity(self, verbosity: int):
        assert isinstance(verbosity, int)
        assert verbosity >= 0
        self._verbosity = verbosity

    @random_state.setter
    def random_state(self, random_state: int):
        assert random_state is None or random_state >= 0
        self._random_state = random_state

    @num_configs_saved.setter
    def num_configs_saved(self, num_configs_saved: int):
        assert num_configs_saved > 0
        self._num_configs_saved = num_configs_saved

    @new_nn_architecture.setter
    def new_nn_architecture(self, new_nn_architecture: bool) -> bool:
        self._new_nn_architecture = new_nn_architecture

    @cuda_visible_devices.setter
    def cuda_visible_devices(self, cuda_visible_devices: bool):
        self._cuda_visible_devices = cuda_visible_devices


# Initialization function for global settings
def init(problem_type, **kwargs):
    """
    Initialize pyMAISE global settings.

    Parameters
    ----------
    problem_type: pyMAISE.ProblemType, {'regression', 'classification'}, or {0, 1}
        Defines a regression or classification problem.
    verbosity: int, default=0
        Level of output.
    random_state: int or None, default=None
        Controls the randomness of all processes in pyMAISE.
    num_configs_saved: int, default=5
        Number of top hyperparameter configurations saved for each
        type of model.
    new_nn_architecture: bool, default=True
        Controls the hyperparameter tuning architecture used for
        tuning neural network models.
    cuda_visible_devices: str or None, default=None
        Devices visible to tensorflow. Sets the CUDA_VISIBLE_DEVICES
        environment variable.

    Returns
    -------
    values: pyMAISE.Settings
        The settings class with the provided parameters changed.
    """
    global values
    values = Settings(problem_type, **kwargs)
    return values
