import copy
import re
import warnings

import numpy as np
import torch
import torch.nn as nn
from skorch import NeuralNetRegressor
from skorch.dataset import ValidSplit

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
from pyMAISE.methods.nn._utils import get_criterion
from pyMAISE.utils.hyperparameters import Choice, HyperParameters

_CLASSIFICATION_LOSSES = {
    "binary_crossentropy",
    "categorical_crossentropy",
    "sparse_categorical_crossentropy",
}


class _SequentialNet(nn.Module):
    """Thin nn.Sequential wrapper so skorch receives a Module subclass."""

    def __init__(self, layers):
        super().__init__()
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class _Float32NeuralNetRegressor(NeuralNetRegressor):
    """NeuralNetRegressor that casts all inputs to float32 before inference.

    xarray / numpy data defaults to float64; PyTorch model weights are float32.
    Overriding infer() here avoids scattering .astype(np.float32) at every call
    site in postprocessor.py and nn_tuner.py.
    """

    def infer(self, x, **fit_params):
        if isinstance(x, torch.Tensor):
            x = x.float()
        else:
            x = torch.tensor(np.asarray(x), dtype=torch.float32)
        return super().infer(x, **fit_params)


class _RecordingTrial:
    """Pseudo-Optuna trial used by get_search_space() to discover hyperparameters.

    Intercepts every suggest_* call made during build() and records the parameter
    name alongside all valid values.  The collected space dict is passed to Optuna's
    GridSampler, which unlike RandomSampler/TPESampler requires the complete search
    space to be declared before any trials run.  This class is never stored or exposed
    to the user — it exists solely so that get_search_space() can introspect the model
    without a live Optuna study.
    """

    def __init__(self):
        self.space = {}

    def _record(self, name, values):
        self.space[name] = values
        return values[0]

    def suggest_int(self, name, low, high, step=1, log=False):
        return self._record(name, list(range(low, high + 1, step)))

    def suggest_float(self, name, low, high, step=None, log=False):
        if step is not None:
            n_steps = round((high - low) / step)
            values = [low + i * step for i in range(n_steps + 1)]
        else:
            # GridSampler requires explicit values; surface both endpoints
            values = [low, high]
        return self._record(name, values)

    def suggest_categorical(self, name, choices):
        return self._record(name, list(choices))


class nnHyperModel:
    # Dictionary of supported layers
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
        "SGD": torch.optim.SGD,
        "RMSprop": torch.optim.RMSprop,
        "Adam": torch.optim.Adam,
        "AdamW": torch.optim.AdamW,
        "Adadelta": torch.optim.Adadelta,
        "Adagrad": torch.optim.Adagrad,
        "Adamax": torch.optim.Adamax,
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

        # Training history; populated by fit()
        self._history = {"loss": [], "val_loss": []}

    # ==========================================================================
    # Methods
    def build(self, trial):
        # Build ordered list of nn.Module layers by walking the architecture tree
        layers = []
        # Conv layers use channels-first layout (N, C, H, W) so in_size must be
        # the channel count = input_shape[0].  Dense/LSTM/GRU networks receive
        # flat or (timesteps, features) inputs where the last dim is the feature
        # count = input_shape[-1].  Inspect the first structural layer to decide.
        first_layer_name = next(iter(self._structural_params))
        if any(k in first_layer_name for k in ("Conv1D", "Conv2D", "Conv3D")):
            in_size = self._input_shape[0]
        else:
            in_size = self._input_shape[-1]
        for layer_name in self._structural_params:
            in_size = self._build_tree(
                layers, layer_name, self._structural_params[layer_name], trial, in_size
            )

        net = _SequentialNet(layers)

        # Determine training criterion from compile_params loss string
        loss = self._compilation_params.get("loss", "mse")
        criterion = get_criterion(loss)

        # Get optimizer class and sampled kwargs
        optim_class, optim_kwargs = self._get_optimizer(trial)

        # Sample fitting params; raises RuntimeError if epochs/batch_size are absent
        fitting = self._sample_fitting_params(trial)
        epochs = fitting["epochs"]
        batch_size = fitting["batch_size"]
        val_split = fitting.get("validation_split", 0.0)

        # Always use NeuralNetRegressor so that predict() returns the raw
        # network output (probabilities/values) rather than class indices.
        # NeuralNetClassifier.predict() would argmax internally, breaking
        # determine_class_from_probabilities() in the CV scoring path.
        net_class = _Float32NeuralNetRegressor

        skorch_kwargs = {
            "module": net,
            "criterion": criterion,
            "optimizer": optim_class,
            "max_epochs": epochs,
            "batch_size": batch_size,
            "train_split": (
                ValidSplit(cv=val_split, stratified=False) if val_split > 0 else None
            ),
            "verbose": 0,
        }
        skorch_kwargs.update({f"optimizer__{k}": v for k, v in optim_kwargs.items()})

        return net_class(**skorch_kwargs)

    def _build_tree(self, layers, layer_name, structural_params, trial, in_size):
        # Get layer object
        layer = copy.deepcopy(self._get_layer(layer_name, structural_params))

        # Run through all number of layers
        for _ in range(layer.num_layers(trial)):
            # Check if there's a wrapper (TimeDistributed, Bidirectional)
            wrapper_data = layer.wrapper()
            if wrapper_data is not None:
                warnings.warn(
                    "Layer wrappers (e.g. TimeDistributed, Bidirectional) are not "
                    "supported in the PyTorch backend and will be ignored.",
                    UserWarning,
                    stacklevel=2,
                )

            module, out_size = layer.build(trial, in_size)
            layers.append(module)
            if out_size is not None:
                in_size = out_size

            # Check for a sublayer
            sublayer_data = layer.sublayer(trial)
            if sublayer_data is not None and sublayer_data[1]:
                in_size = self._build_tree(
                    layers, sublayer_data[0], sublayer_data[1], trial, in_size
                )

            # Increment current layer
            layer.increment_layer()

        # Reset the layer object
        layer.reset()

        return in_size

    def fit(self, trial, model, x, y):
        x_t = torch.tensor(np.asarray(x), dtype=torch.float32)

        # CrossEntropyLoss requires (N,) long integer class indices, not floats.
        # Convert one-hot (N, C) arrays via argmax and integer (N, 1) arrays via
        # squeeze so that both label formats arrive in the shape PyTorch expects.
        loss = self._compilation_params.get("loss", "mse")
        if loss in ("categorical_crossentropy", "sparse_categorical_crossentropy"):
            y_np = np.asarray(y)
            if y_np.ndim > 1 and y_np.shape[-1] > 1:
                y_t = torch.tensor(np.argmax(y_np, axis=-1), dtype=torch.long)
            else:
                y_t = torch.tensor(y_np.ravel().astype(np.int64), dtype=torch.long)
        else:
            y_t = torch.tensor(np.asarray(y), dtype=torch.float32)

        model.fit(x_t, y_t)

        train_loss = list(model.history[:, "train_loss"])
        try:
            val_loss = list(model.history[:, "valid_loss"])
        except KeyError:
            val_loss = []

        self._history = {"loss": train_loss, "val_loss": val_loss}
        return self._history

    def _sample_fitting_params(self, trial):
        # Sample HyperParameters values; validate required keys are present
        sampled = copy.deepcopy(self._fitting_params)
        for key, value in sampled.items():
            if isinstance(value, HyperParameters):
                sampled[key] = value.sample(trial, key)

        if "epochs" not in sampled or sampled["epochs"] is None:
            raise RuntimeError(
                "fitting_params must include 'epochs'. "
                "Example: fitting_params={'epochs': 50, 'batch_size': 32}"
            )
        if "batch_size" not in sampled or sampled["batch_size"] is None:
            raise RuntimeError(
                "fitting_params must include 'batch_size'. "
                "Example: fitting_params={'epochs': 50, 'batch_size': 32}"
            )

        if sampled.get("callbacks"):
            warnings.warn(
                "fitting_params['callbacks'] is not supported in the PyTorch backend "
                "and will be ignored. Use Optuna callbacks via the study API instead.",
                UserWarning,
                stacklevel=2,
            )
            del sampled["callbacks"]

        # During hyperparameter search the outer CV fold acts as validation, so
        # an inner split would waste training data.  Default to 0 (no split).
        sampled.setdefault("validation_split", 0.0)

        return sampled

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

    def _get_optimizer(self, trial):
        # Get optimizer name
        optimizer_name = copy.deepcopy(self._optimizer)
        if isinstance(self._optimizer, Choice):
            optimizer_name = optimizer_name.sample(trial, "optimizer")

        # Make sure the optimizer parameters were given by the user
        assert self._optimizer_params[optimizer_name]

        # Copy data and sample hyperparameters
        sampled = copy.deepcopy(self._optimizer_params[optimizer_name])
        for key, value in sampled.items():
            if isinstance(value, HyperParameters):
                sampled[key] = value.sample(trial, "_".join([optimizer_name, key]))

        # Translate Keras-style key to PyTorch
        if "learning_rate" in sampled:
            sampled["lr"] = sampled.pop("learning_rate")

        # Search for supported optimizer
        if optimizer_name in self.optimizer_dict:
            return self.optimizer_dict[optimizer_name], sampled

        # If the optimizer name doesn't exist in supported optimizer
        # dictionary throw error
        raise RuntimeError(f"Optimizer ({optimizer_name}) is not supported")

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

    def get_search_space(self):
        """Return the hyperparameter search space dict required by Optuna's GridSampler.

        Runs build() once with a _RecordingTrial to capture every suggest_* call
        the model makes, then returns the collected {name: [values]} dict.
        """
        recorder = _RecordingTrial()
        self.build(recorder)
        return recorder.space
