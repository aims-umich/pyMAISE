import torch.nn as nn

from pyMAISE.methods.nn._layer import Layer
from pyMAISE.methods.nn._utils import get_activation


class _Conv2DBlock(nn.Module):
    """2-D convolutional layer with optional activation.

    Note: PyTorch Conv2d uses channels-first layout (N, C, H, W).
    Input data should be shaped accordingly.
    """

    def __init__(self, in_channels, filters, kernel_size, stride, padding, activation, use_bias):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, filters, kernel_size,
            stride=stride, padding=padding, bias=use_bias,
        )
        self.activation = get_activation(activation)

    def forward(self, x):
        return self.activation(self.conv(x))


def _resolve_padding(padding_str, kernel_size):
    if padding_str == "same":
        if isinstance(kernel_size, (list, tuple)):
            return tuple(k // 2 for k in kernel_size)
        return kernel_size // 2
    return 0  # "valid"


class Conv2DLayer(Layer):
    def __init__(self, layer_name, parameters: dict):
        # Initialize layer data
        self.reset()
        super().__init__(layer_name, parameters)

        # Build layer data
        self._data = super().build_data(self._data, parameters)

        # Assert non-default variables are defined
        assert self._data["filters"] is not None
        assert self._data["kernel_size"] is not None

    # ==========================================================================
    # Methods
    def build(self, trial, in_size):
        # Sample parameters and build PyTorch Conv2d module.
        params = super().sample_parameters(self._data, trial)
        filters = params["filters"]
        kernel_size = params["kernel_size"]
        stride = params.get("strides", (1, 1))
        padding = _resolve_padding(params.get("padding", "valid"), kernel_size)
        activation = params.get("activation", "None")
        use_bias = params.get("use_bias", True)
        return _Conv2DBlock(in_size, filters, kernel_size, stride, padding, activation, use_bias), filters

    def reset(self):
        self._data = {
            "filters": None,
            "kernel_size": None,
            "strides": (1, 1),
            "padding": "valid",
            "activation": "None",
            "use_bias": True,
        }
        super().reset()

    def increment_layer(self):
        return super().increment_layer()

    # ==========================================================================
    # Getters
    def num_layers(self, trial):
        return super().num_layers(trial)

    def sublayer(self, trial):
        return super().sublayer(trial)

    def wrapper(self):
        return super().wrapper()
