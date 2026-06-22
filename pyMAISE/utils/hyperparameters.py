import numpy as np


class HyperParameters:
    def __init__(self, default=None, parent_name=None, parent_values=None):
        self._default = default
        self._parent_name = parent_name
        self._parent_values = parent_values


class Boolean(HyperParameters):
    """
    Define a boolean hyperparameter for neural network hyperparameter tuning.
    """

    def __init__(self, default=None, parent_name=None, parent_values=None):
        HyperParameters.__init__(self, default, parent_name, parent_values)

    def sample(self, trial, hp_name):
        """Sample this hyperparameter from an Optuna trial."""
        return trial.suggest_categorical(name=hp_name, choices=[True, False])

    def grid_values(self):
        return [True, False]


class Int(HyperParameters):
    """
    Define an integer hyperparameter for neural network hyperparameter tuning.

    Parameters
    ----------
    min_value: int
        Minimum value (inclusive).
    max_value: int
        Maximum value (inclusive).
    step: int or None, default=None
        Step between values. Defaults to 1.
    sampling: {'linear', 'log', 'reverse_log'}, default='linear'
        Distribution for sampling. Use 'log' for log-scale sampling.
    default: int or None, default=None
        Default value.
    """

    def __init__(
        self,
        min_value,
        max_value,
        step=None,
        sampling="linear",
        default=None,
        parent_name=None,
        parent_values=None,
    ):
        self._min_value = min_value
        self._max_value = max_value
        self._step = step
        self._sampling = sampling

        HyperParameters.__init__(self, default, parent_name, parent_values)

    def sample(self, trial, hp_name):
        """Sample this hyperparameter from an Optuna trial."""
        log = self._sampling == "log"
        return trial.suggest_int(
            name=hp_name,
            low=self._min_value,
            high=self._max_value,
            step=self._step or 1,
            log=log,
        )

    def grid_values(self):
        step = self._step or 1
        return list(range(self._min_value, self._max_value + 1, step))


class Float(HyperParameters):
    """
    Define a floating point hyperparameter for neural network hyperparameter tuning.

    Parameters
    ----------
    min_value: float
        Minimum value (inclusive).
    max_value: float
        Maximum value (inclusive).
    step: float or None, default=None
        Step between values. Required for grid search.
    sampling: {'linear', 'log', 'reverse_log'}, default='linear'
        Distribution for sampling. Use 'log' for log-scale sampling.
    default: float or None, default=None
        Default value.
    """

    def __init__(
        self,
        min_value,
        max_value,
        step=None,
        sampling="linear",
        default=None,
        parent_name=None,
        parent_values=None,
    ):
        self._min_value = min_value
        self._max_value = max_value
        self._step = step
        self._sampling = sampling

        HyperParameters.__init__(self, default, parent_name, parent_values)

    def sample(self, trial, hp_name):
        """Sample this hyperparameter from an Optuna trial."""
        log = self._sampling == "log"
        return trial.suggest_float(
            name=hp_name,
            low=self._min_value,
            high=self._max_value,
            step=self._step,
            log=log,
        )

    def grid_values(self):
        if self._step is None:
            raise ValueError(
                f"Float hyperparameter requires a 'step' for grid search. "
                f"Use Choice([...]) for discrete float values."
            )
        return list(np.arange(self._min_value, self._max_value + self._step / 2, self._step))


class Choice(HyperParameters):
    """
    Define a categorical choice hyperparameter for neural network hyperparameter tuning.

    Parameters
    ----------
    values: list
        The possible choices (strings, ints, floats, or bools).
    """

    def __init__(
        self, values, ordered=None, default=None, parent_name=None, parent_values=None
    ):
        self._values = values
        self._ordered = ordered

        HyperParameters.__init__(self, default, parent_name, parent_values)

    def sample(self, trial, hp_name):
        """Sample this hyperparameter from an Optuna trial."""
        return trial.suggest_categorical(name=hp_name, choices=self._values)

    def grid_values(self):
        return list(self._values)

    @property
    def values(self):
        """list: The possible choices for this hyperparameter."""
        return self._values


class Fixed(HyperParameters):
    """
    Define a fixed (non-tunable) hyperparameter.

    Parameters
    ----------
    value: any
        The fixed value.
    """

    def __init__(self, value, parent_name=None, parent_values=None):
        self._value = value

        HyperParameters.__init__(
            self, parent_name=parent_name, parent_values=parent_values
        )

    def sample(self, trial, hp_name):
        """Return the fixed value (no sampling)."""
        return self._value

    def grid_values(self):
        return [self._value]
