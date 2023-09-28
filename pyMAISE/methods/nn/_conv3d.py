import copy

from keras.layers import Conv3D

from pyMAISE.utils.hyperparameters import HyperParameters


# Dense layer keras neural networks
class Conv3dLayer:
    def __init__(self, layer_name, parameters: dict):
        self._layer_name = layer_name
        self.reset()

        for key, value in parameters.items():
            if key == "num_layers":
                self._num_layers = value
                continue

            self._data[key] = value

        # Assert keras non-default variables are defined
        assert self._data["filters"] != None

    # ==========================================================================
    # Methods
    def build(self, hp):
        # Set pyMAISE hyperparameter to keras-tuner hyperparameter
        sampled_data = copy.deepcopy(self._data)
        for key, value in self._data.items():
            if isinstance(value, HyperParameters):
                assert self._data[key]
                sampled_data[key] = value.hp(
                    hp, "_".join([self._layer_name + str(self._current_layer), key])
                )
            else:
                sampled_data[key] = value

        return Conv3D(**sampled_data)

    def reset(self):
        self._current_layer = 0
        self._num_layers = 1
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

    def increment_layer(self):
        self._current_layer = self._current_layer + 1

    # ==========================================================================
    # Getters
    def num_layers(self, hp):
        if isinstance(self._num_layers, HyperParameters):
            return self._num_layers.hp(hp, self._layer_name + "_num_layers")
        else:
            return self._num_layers
