import copy
import re

from keras_tuner import HyperModel
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import (
    SGD,
    Adadelta,
    Adafactor,
    Adagrad,
    Adam,
    Adamax,
    AdamW,
    Ftrl,
    Nadam,
    RMSprop,
)

import pyMAISE.settings as settings
from pyMAISE.methods.nn._conv1d import Conv1DLayer
from pyMAISE.methods.nn._conv2d import Conv2DLayer
from pyMAISE.methods.nn._conv3d import Conv3DLayer
from pyMAISE.methods.nn._dense import DenseLayer
from pyMAISE.methods.nn._dropout import DropoutLayer
from pyMAISE.methods.nn._flatten import FlattenLayer
from pyMAISE.methods.nn._gru import GRULayer
from pyMAISE.methods.nn._lstm import LSTMLayer
from pyMAISE.methods.nn._max_pooling_1d import MaxPooling1DLayer
from pyMAISE.methods.nn._max_pooling_2d import MaxPooling2DLayer
from pyMAISE.methods.nn._max_pooling_3d import MaxPooling3DLayer
from pyMAISE.methods.nn._reshape import ReshapeLayer
from pyMAISE.utils.hyperparameters import Choice, HyperParameters


class nnHyperModel(HyperModel):
    # Dictionary of supported Layers
    layer_dict = {
        "Dense": DenseLayer,
        "Dropout": DropoutLayer,
        "LSTM": LSTMLayer,
        "GRU": GRULayer,
        "Conv1D": Conv1DLayer,
        "Conv2D": Conv2DLayer,
        "Conv3D": Conv3DLayer,
        "MaxPooling1D": MaxPooling1DLayer,
        "MaxPooling2D": MaxPooling2DLayer,
        "MaxPooling3D": MaxPooling3DLayer,
        "Flatten": FlattenLayer,
        "Reshape": ReshapeLayer,
    }

    # Dictionary of supported optimizers
    optimizer_dict = {
        "SGD": SGD,
        "RMSprop": RMSprop,
        "Adam": Adam,
        "AdamW": AdamW,
        "Adadelta": Adadelta,
        "Adagrad": Adagrad,
        "Adamax": Adamax,
        "Adafactor": Adafactor,
        "Nadam": Nadam,
        "Ftrl": Ftrl,
    }

    def __init__(self, parameters: dict, input_shape, name):
        # Structure/Architectural hyperparameters
        self._structural_params = parameters["structural_params"]

        # Optimizers and their hyperparameters
        if parameters["optimizer"]:
            self._optimizer = parameters["optimizer"]

            self._optimizer_params = {}

            if isinstance(self._optimizer, Choice):
                for optimizer in self._optimizer.values:
                    assert parameters[optimizer]
                    self._optimizer_params[optimizer] = parameters[optimizer]
            else:
                assert parameters[self._optimizer]
                self._optimizer_params[self._optimizer] = parameters[self._optimizer]
        else:
            raise RuntimeError("Optimizer was not given in `optimizer` key")

        # Model compilation hyperparameters
        self._compilation_params = parameters["compile_params"]

        # Model fitting hyperparameters
        self._fitting_params = parameters["fitting_params"]

        # Input data shape
        self._input_shape = input_shape

        # Model name
        self._name = name

    # ==========================================================================
    # Methods
    def build(self, hp):
        # Sequential keras neural network
        model = Sequential(name=self._name)

        # Add input layer
        model.add(Input(shape=self._input_shape, name=self._name + "_Input"))

        # Iterating though archetecture
        for layer_name in self._structural_params.keys():
            model = self._build_tree(
                model, layer_name, self._structural_params[layer_name], hp
            )

        # Compile Model
        self._compilation_params["optimizer"] = self._get_optimizer(hp)
        model.compile(**self._compilation_params)
        return model

    def _build_tree(self, model, layer_name, structural_params, hp):
        # Get layer object
        layer = copy.deepcopy(self._get_layer(layer_name, structural_params))

        # Run through all number of layers
        for _ in range(layer.num_layers(hp)):
            # Check if there's a wrapper (TimeDistributed, Bidirectional)
            wrapper_data = layer.wrapper()
            if wrapper_data is not None:
                model.add(wrapper_data[0](layer.build(hp), **wrapper_data[1]))
            else:
                model.add(layer.build(hp))

            # Check for a sublayer
            sublayer_data = layer.sublayer(hp)
            if sublayer_data is not None and sublayer_data[1]:
                model = self._build_tree(model, sublayer_data[0], sublayer_data[1], hp)

            # Increment current layer
            layer.increment_layer()

        # Reset the layer object
        layer.reset()

        return model

    # Fit function for keras-tuner to allow hyperparameter tuning of fitting parameters
    def fit(self, hp, model, x, y, **kwargs):
        fitting_params = copy.deepcopy(self._fitting_params)
        for key, value in self._fitting_params.items():
            if isinstance(value, HyperParameters):
                fitting_params[key] = value.hp(hp, key)

        return model.fit(
            x,
            y,
            **fitting_params,
            verbose=settings.values.verbosity,
            **kwargs,
        )

    # Update parameters after tuning, a common use case is increasing
    # the number of epochs
    def set_params(self, parameters: dict = None):
        if "structural_params" in parameters:
            for key, value in parameters["structural_params"].items():
                assert key in self._structural_params
                for param_key, param_value in value.items():
                    self._structural_params[key][param_key] = param_value

        elif "optimizer" in parameters:
            self._optimizer = parameters["optimizer"]
            if parameters[self._optimizer]:
                for key, value in parameters[self._optimizer].items():
                    self._optimizer_params[self._optimizer][key] = value

        elif "compile_params" in parameters:
            for key, value in parameters["compile_params"].items():
                self._compilation_params[key] = value

        elif "fitting_params" in parameters:
            for key, value in parameters["fitting_params"].items():
                self._fitting_params[key] = value

    def _get_layer(self, layer_name, structural_params):
        # Search through supported layers dictionary to find layer,
        # if multiple then take the first as the layer
        layer = None
        position = None
        for key, value in self.layer_dict.items():
            match_idx = re.search(key, layer_name)
            if match_idx is not None and (
                position is None or match_idx.span()[0] > position
            ):
                layer = value
                position = match_idx.span()[0]

        if layer is not None:
            return layer(layer_name, structural_params)
        else:
            # If not found we throw an error
            raise RuntimeError(f"Layer ({layer_name}) is not supported")

    def _get_optimizer(self, hp):
        # Get optimizer name
        optimizer = copy.deepcopy(self._optimizer)
        if isinstance(self._optimizer, Choice):
            optimizer = optimizer.hp(hp, "optimizer")

        # Make sure the optimizer parameters were given by the user
        assert self._optimizer_params[optimizer]

        # Copy data and sample hyperparameters
        sampled_data = copy.deepcopy(self._optimizer_params[optimizer])
        for key, value in sampled_data.items():
            if isinstance(value, HyperParameters):
                sampled_data[key] = value.hp(hp, "_".join([optimizer, key]))

        # Search for support optimizer
        if optimizer in self.optimizer_dict:
            return self.optimizer_dict[optimizer](**sampled_data)

        # If the optimizer name doesn't exit in supported optimizer
        # dictionary throw error
        raise RuntimeError(f"Optimizer ({optimizer}) is not supported")

    def get_hyperparameters(self):
        hps = []

        def search_dict(d):
            for _, v in d.items():
                if isinstance(v, HyperParameters):
                    hps.append(v)

                elif isinstance(v, dict):
                    search_dict(v)

        for d in [
            self._structural_params,
            self._compilation_params,
            self._fitting_params,
            self._optimizer_params,
        ]:
            search_dict(d)

        return hps
