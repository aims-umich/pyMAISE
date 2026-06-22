import torch.nn as nn

from pyMAISE.methods.nn._layer import Layer


class FlattenLayer(Layer):
    def __init__(self, layer_name, parameters: dict):
        # Initialize layer and base class
        self.reset()
        super().__init__(layer_name, parameters)

        # Get layer data from params dictionary
        self._data = super().build_data(self._data, parameters)

    # ==========================================================================
    # Methods
    def build(self, trial, in_size):
        # Return -1 as the sentinel out_size so that the next Dense layer
        # knows to use LazyLinear (the actual flat size is unknown at build
        # time because it depends on the spatial dimensions of the Conv output).
        return nn.Flatten(), -1

    def reset(self):
        self._data = {}
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
