import torch.nn as nn

from pyMAISE.methods.nn._layer import Layer


class DropoutLayer(Layer):
    def __init__(self, layer_name, parameters: dict):
        # Initialize layer data
        self.reset()
        super().__init__(layer_name, parameters)

        # Build layer data
        self._data = super().build_data(self._data, parameters)

    # ==========================================================================
    # Methods
    def build(self, trial, in_size):
        # Sample parameters and build PyTorch Dropout module
        params = super().sample_parameters(self._data, trial)
        return nn.Dropout(p=params["rate"]), in_size

    def reset(self):
        self._data = {
            "rate": 0.2,
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
