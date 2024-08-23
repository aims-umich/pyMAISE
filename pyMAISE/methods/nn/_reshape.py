from tensorflow.keras.layers import Reshape

from pyMAISE.methods.nn._layer import Layer


class ReshapeLayer(Layer):
    def __init__(self, layer_name, parameters: dict):
        # Initialize layer and base class
        self.reset()
        super().__init__(layer_name, parameters)

        # Get layer data from params dictionary
        self._data = super().build_data(self._data, parameters)

        # Assert keras non-default variables are defined
        assert self._data["target_shape"] is not None

    # ==========================================================================
    # Methods
    def build(self, hp):
        return Reshape(**super().sample_parameters(self._data, hp))

    def reset(self):
        self._data = {
            "target_shape": None,
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
