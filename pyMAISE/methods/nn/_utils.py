import torch.nn as nn
import torch.nn.functional as F

# Maps Keras-style activation strings to PyTorch nn.Module constructors.
ACTIVATION_MAP = {
    None: nn.Identity,
    "None": nn.Identity,
    "none": nn.Identity,
    "linear": nn.Identity,
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "softmax": lambda: nn.Softmax(dim=-1),
    "elu": nn.ELU,
    "selu": nn.SELU,
    "gelu": nn.GELU,
    "leaky_relu": nn.LeakyReLU,
    "prelu": nn.PReLU,
    "hardswish": nn.Hardswish,
    "mish": nn.Mish,
}


def validate_output_shape(y_pred, y_true):
    """
    When using NLL as the loss function, the output layer must be **double** the size of the
    number of outputs. If the shape of the predictions do not match this, raise an error so
    that other systems do not fail unexpectedly.
    """
    if y_pred.shape[-1] != 2 * y_true.shape[-1]:
        raise ValueError(
            f"Gaussian NLL loss expects the network's final output dimension ({y_pred.shape[-1]}) "
            f"to be exactly double the target dimension ({y_true.shape[-1]} * 2 = {2 * y_true.shape[-1]}). "
            f"Please update your final output layer 'units' to match this."
        )


def split_mean_var(output):
    """
    Splits a (..., 2*n_targets) tensor into mean and variance halves for
    heteroscedastic (NLL) models, where the network's last layer produces
    [mean | raw_variance] concatenated along the last axis.

    Softplus is applied to the raw variance half so it's always positive --
    nn.GaussianNLLLoss raises on non-positive variance, and the network's
    raw linear output is otherwise unconstrained. A small epsilon is added
    for numerical stability.

    Shared by _GaussianNLLCriterion (training) and
    DeepEnsemble.predict_with_uncertainty (inference) so both always agree
    on how variance is derived from the raw network output.
    """
    n_targets = output.shape[-1] // 2
    mean = output[..., :n_targets]
    raw_var = output[..., n_targets:]
    var = F.softplus(raw_var) + 1e-6
    return mean, var


class _GaussianNLLCriterion(nn.Module):
    """
    Wraps nn.GaussianNLLLoss for a single concatenated network output of
    shape (batch, 2*n_targets) = [mean | raw_variance].

    Needed because skorch calls criterion(y_pred, y_true) with two
    arguments, while nn.GaussianNLLLoss.forward expects three:
    (mean, target, var).
    """

    def __init__(self):
        super().__init__()
        self._nll = nn.GaussianNLLLoss()

    def forward(self, y_pred, y_true):
        validate_output_shape(y_pred, y_true)
        mean, var = split_mean_var(y_pred)
        return self._nll(mean, y_true, var)


# Maps Keras-style loss strings to PyTorch criterion classes.
# compile_params["loss"] is the training criterion; compile_params["metrics"]
# is ignored — PostProcessor computes evaluation metrics separately.
CRITERION_MAP = {
    "mse": nn.MSELoss,
    "mean_squared_error": nn.MSELoss,
    "mae": nn.L1Loss,
    "mean_absolute_error": nn.L1Loss,
    "binary_crossentropy": nn.BCEWithLogitsLoss,
    "categorical_crossentropy": nn.CrossEntropyLoss,
    "sparse_categorical_crossentropy": nn.CrossEntropyLoss,
    "huber": nn.HuberLoss,
    "huber_loss": nn.HuberLoss,
    "nll": _GaussianNLLCriterion,
}


def get_activation(name):
    """Return an instantiated PyTorch activation module for the given name string."""
    if name not in ACTIVATION_MAP:
        raise ValueError(
            f"Unsupported activation '{name}'. "
            f"Supported: {list(ACTIVATION_MAP.keys())}"
        )
    factory = ACTIVATION_MAP[name]
    return factory()


def get_criterion(name):
    """Return the PyTorch criterion class for the given Keras-style loss name."""
    if name not in CRITERION_MAP:
        raise ValueError(
            f"Unsupported loss '{name}'. "
            f"Supported: {list(CRITERION_MAP.keys())}"
        )
    return CRITERION_MAP[name]

