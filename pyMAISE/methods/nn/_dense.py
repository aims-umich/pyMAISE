import torch.nn as nn

from pyMAISE.methods.nn._layer import Layer
from pyMAISE.methods.nn._utils import get_activation


class _DenseBlock(nn.Module):
    """Dense (fully-connected) layer with optional activation.

    Uses ``nn.LazyLinear`` when ``in_features`` is unknown (e.g. immediately
    after a ``Flatten`` layer whose flat size depends on runtime spatial dims).
    ``in_features <= 0`` is the sentinel produced by ``FlattenLayer.build()``.
    """

    def __init__(self, in_features, units, activation, use_bias):
        super().__init__()
        if in_features is None or in_features <= 0:
            self.linear = nn.LazyLinear(units, bias=use_bias)
        else:
            self.linear = nn.Linear(in_features, units, bias=use_bias)
        self.activation = get_activation(activation)

    def forward(self, x):
        return self.activation(self.linear(x))


class DenseLayer(Layer):
    def __init__(self, layer_name, parameters: dict):
        # Initialize layer data
        self.reset()
        super().__init__(layer_name, parameters)

        # Build layer data
        self._data = super().build_data(self._data, parameters)

        assert self._data["units"] is not None

    # ==========================================================================
    # Methods
    def build(self, trial, in_size):
        # Sample parameters and build PyTorch module
        params = super().sample_parameters(self._data, trial)
        units = params["units"]
        activation = params.get("activation")
        use_bias = params.get("use_bias", True)
        return _DenseBlock(in_size, units, activation, use_bias), units

    def reset(self):
        self._data = {
            "units": None,
            "activation": None,
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
