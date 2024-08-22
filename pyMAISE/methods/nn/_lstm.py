from tensorflow.keras.layers import LSTM

from pyMAISE.methods.nn._layer import Layer


class LSTMLayer(Layer):
    def __init__(self, layer_name, parameters: dict):
        # Initialize layer and base class
        self.reset()
        super().__init__(layer_name, parameters)

        # Get layer data from params dictionary
        self._data = super().build_data(self._data, parameters)

        # Assert keras non-default variables are defined
        assert self._data["units"] is not None

    # ==========================================================================
    # Methods
    def build(self, hp):
        # Set pyMAISE hyperparameter to keras-tuner hyperparameter
        return LSTM(**super().sample_parameters(self._data, hp))

    def reset(self):
        self._data = {
            "units": None,
            "activation": "tanh",
            "recurrent_activation": "sigmoid",
            "use_bias": True,
            "kernel_initializer": "glorot_uniform",
            "recurrent_initializer": "orthogonal",
            "bias_initializer": "zeros",
            "unit_forget_bias": True,
            "kernel_regularizer": None,
            "recurrent_regularizer": None,
            "bias_regularizer": None,
            "activity_regularizer": None,
            "kernel_constraint": None,
            "recurrent_constraint": None,
            "bias_constraint": None,
            "dropout": 0.0,
            "recurrent_dropout": 0.0,
            "return_sequences": False,
            "return_state": False,
            "go_backwards": False,
            "stateful": False,
            "unroll": False,
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
