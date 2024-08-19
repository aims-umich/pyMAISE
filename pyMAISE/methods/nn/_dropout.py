from tensorflow.keras.layers import Dropout

import pyMAISE.settings as settings
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
    def build(self, hp):
        # Set pyMAISE hyperparameter to keras-tuner hyperparameter
        return Dropout(**super().sample_parameters(self._data, hp))

    def reset(self):
        self._data = {
            "rate": 0.2,
            "noise_shape": None,
            "seed": settings.values.random_state,
        }
        super().reset()

    def increment_layer(self):
        return super().increment_layer()

    # ==========================================================================
    # Getters
    def num_layers(self, hp):
        return super().num_layers(hp)

    def sublayer(self, hp):
        return super().sublayer(hp)

    def wrapper(self):
        return super().wrapper()
