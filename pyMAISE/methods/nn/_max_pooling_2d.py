import torch.nn as nn

from pyMAISE.methods.nn._layer import Layer


class MaxPooling2DLayer(Layer):
    def __init__(self, layer_name, parameters: dict):
        # Initialize layer data
        self.reset()
        super().__init__(layer_name, parameters)

        # Build layer data
        self._data = super().build_data(self._data, parameters)

    # ==========================================================================
    # Methods
    def build(self, trial, in_size):
        # Sample parameters and build PyTorch MaxPool2d module.
        params = super().sample_parameters(self._data, trial)
        pool_size = params["pool_size"]
        strides = params.get("strides") or pool_size
        return nn.MaxPool2d(kernel_size=pool_size, stride=strides), in_size

    def reset(self):
        self._data = {
            "pool_size": (2, 2),
            "strides": None,
            "padding": "valid",  # kept for API compat
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
