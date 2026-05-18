import torch.nn as nn

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
