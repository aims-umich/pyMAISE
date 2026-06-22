import torch.nn as nn

from pyMAISE.methods.nn._layer import Layer


class _ReshapeBlock(nn.Module):
    """Reshape all dims after the batch dimension to target_shape."""

    def __init__(self, target_shape):
        super().__init__()
        self.target_shape = target_shape

    def forward(self, x):
        return x.view(x.shape[0], *self.target_shape)


class ReshapeLayer(Layer):
    def __init__(self, layer_name, parameters: dict):
        # Initialize layer and base class
        self.reset()
        super().__init__(layer_name, parameters)

        # Get layer data from params dictionary
        self._data = super().build_data(self._data, parameters)

        # Assert non-default variables are defined
        assert self._data["target_shape"] is not None

    # ==========================================================================
    # Methods
    def build(self, trial, in_size):
        # Sample parameters and build PyTorch reshape module.
        # out_size is None because flat size after reshape is context-dependent.
        params = super().sample_parameters(self._data, trial)
        target_shape = params["target_shape"]
        return _ReshapeBlock(target_shape), None

    def reset(self):
        self._data = {
            "target_shape": None,
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
