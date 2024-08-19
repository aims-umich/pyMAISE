from tensorflow.keras.layers import Conv3D

from pyMAISE.methods.nn._layer import Layer


class Conv3DLayer(Layer):
    def __init__(self, layer_name, parameters: dict):
        # Initialize layer data
        self.reset()
        super().__init__(layer_name, parameters)

        # Build layer data
        self._data = super().build_data(self._data, parameters)

        # Assert keras non-default variables are defined
        assert self._data["filters"] is not None
        assert self._data["kernel_size"] is not None

    # ==========================================================================
    # Methods
    def build(self, hp):
        # Set pyMAISE hyperparameter to keras-tuner hyperparameter
        return Conv3D(**super().sample_parameters(self._data, hp))

    def reset(self):
        self._data = {
            "filters": None,
            "kernel_size": None,
            "strides": (1, 1, 1),
            "padding": "valid",
            "data_format": None,
            "dilation_rate": (1, 1, 1),
            "groups": 1,
            "activation": "None",
            "use_bias": True,
            "kernel_initializer": "glorot_uniform",
            "bias_initializer": "zeros",
            "kernel_regularizer": None,
            "bias_regularizer": None,
            "activity_regularizer": None,
            "kernel_constraint": None,
            "bias_constraint": None,
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
