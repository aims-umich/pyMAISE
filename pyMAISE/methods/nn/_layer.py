import copy

from pyMAISE.utils.hyperparameters import Choice, HyperParameters


class Layer:
    def __init__(self, layer_name, parameters):
        # Set layer name and reset the layer
        self._layer_name = layer_name
        self.reset()

        # Set self._base_data values
        for key, value in parameters.items():
            if key == "sublayer":
                # Save the sublayer data and determine if multiple sublayers
                # are given for pyMAISE.Choice
                self._base_data["sublayer"] = value
                sublayer_list = value
                if isinstance(self._base_data["sublayer"], Choice):
                    sublayer_list = value.values
                else:
                    sublayer_list = [sublayer_list]

                # We set sublayer data to none but expect it to be updated on
                # later key, value pairs in the dictionary assuming that they
                # proceed "sublayer"
                for sublayer_name in sublayer_list:
                    self._base_data[sublayer_name] = None

            elif key in self._base_data:
                self._base_data[key] = value

        # Make sure the wrapper is a tuple
        if self._base_data["wrapper"] is not None and not isinstance(
            self._base_data["wrapper"], tuple
        ):
            self._base_data["wrapper"] = (parameters["wrapper"], {})

        # Assert we have a layer name
        assert self._layer_name is not None

    # ==========================================================================
    # Methods
    def build_data(self, data, parameters):
        # Get data for keys shared between data and parameters (or input_shape)
        for key, value in parameters.items():
            if key in data or key == "input_shape" or key == "input_dim":
                data[key] = value
        return data

    def sample_parameters(self, data, trial):
        # Sample hyperparameter data during building
        sampled_data = copy.deepcopy(data)
        for key, value in sampled_data.items():
            if isinstance(value, HyperParameters):
                sampled_data[key] = value.sample(
                    trial,
                    "_".join([f"{self._layer_name}_{self._current_layer}", key]),
                )
        return sampled_data

    def reset(self):
        self._current_layer = 0
        self._base_data = {
            "num_layers": 1,
            "wrapper": None,
            "sublayer": None,
        }

    def increment_layer(self):
        # Increment layer by one
        self._current_layer = self._current_layer + 1

    def num_layers(self, trial):
        # Determine number of layers, if HyperParameters then sample it
        if isinstance(self._base_data["num_layers"], HyperParameters):
            return self._base_data["num_layers"].sample(
                trial, self._layer_name + "_num_layers"
            )
        else:
            return self._base_data["num_layers"]

    def sublayer(self, trial):
        # Get sublayer data, if a Choice then sample
        if isinstance(self._base_data["sublayer"], Choice):
            sublayer_name = self._base_data["sublayer"].sample(
                trial, f"{self._layer_name}_{self._current_layer}_sublayer"
            )
            if sublayer_name != "None":
                return (
                    f"{self._layer_name}_{self._current_layer}"
                    + f"_sublayer_{sublayer_name}",
                    self._base_data[sublayer_name],
                )
        elif isinstance(self._base_data["sublayer"], str):
            sublayer_name = self._base_data["sublayer"]
            return (
                f"{self._layer_name}_{self._current_layer}_sublayer_{sublayer_name}",
                self._base_data[sublayer_name],
            )

        return None

    def wrapper(self):
        # Return wrapper
        return self._base_data["wrapper"]
