import pytest
import torch.nn as nn
import optuna
from skorch import NeuralNetRegressor

import pyMAISE as mai
from pyMAISE import Choice, Int
from pyMAISE.methods import nnHyperModel
from pyMAISE.methods.nn._nn_hypermodel import _SequentialNet
from pyMAISE.methods.nn._dense import _DenseBlock


def test_fnn_build():
    """build() returns an initialised NeuralNetRegressor with the correct
    layer sequence, training params, and output size.
    Keras-only params (kernel_initializer) must be silently ignored."""
    mai.init(problem_type="regression")

    fnn_settings = {
        "structural_params": {
            "Dense_input": {
                "units": Int(min_value=25, max_value=250),
                "activation": "relu",
                "kernel_initializer": "normal",  # Keras-only; silently dropped
                "sublayer": "Dropout",
                "Dropout": {"rate": 0.5},
            },
            "Dense_hidden": {
                "num_layers": 2,
                "units": Int(min_value=25, max_value=250),
                "activation": "relu",
                "kernel_initializer": "normal",
            },
            "Dense_output": {
                "units": 22,
                "activation": "linear",
                "kernel_initializer": "normal",
            },
        },
        "optimizer": "Adam",
        "Adam": {"learning_rate": 0.0001},
        "compile_params": {"loss": "mean_absolute_error"},
        "fitting_params": {
            "batch_size": 16,
            "epochs": 50,
            "validation_split": 0.15,
        },
    }

    hypermodel = nnHyperModel(fnn_settings, input_shape=(6,), name="")

    # Pin every sampled hyperparameter to its minimum value.
    search_space = hypermodel.get_search_space()
    trial = optuna.trial.FixedTrial({k: v[0] for k, v in search_space.items()})
    model = hypermodel.build(trial)
    model.initialize()

    assert isinstance(model, NeuralNetRegressor)
    assert isinstance(model.module_, _SequentialNet)

    # Expected sequence:
    #   Dense_input_0  (_DenseBlock: Linear + ReLU)
    #   Dropout sublayer
    #   Dense_hidden_0 (_DenseBlock)
    #   Dense_hidden_1 (_DenseBlock)
    #   Dense_output_0 (_DenseBlock: Linear + Identity)
    children = list(model.module_.net.children())
    assert len(children) == 5
    assert isinstance(children[0], _DenseBlock)
    assert isinstance(children[1], nn.Dropout)
    assert isinstance(children[2], _DenseBlock)
    assert isinstance(children[3], _DenseBlock)
    assert isinstance(children[4], _DenseBlock)

    assert 25 <= children[0].linear.out_features <= 250  # Dense_input units in range
    assert children[4].linear.out_features == 22         # Dense_output fixed at 22
    assert children[1].p == 0.5                          # Dropout rate

    assert model.max_epochs == 50
    assert model.batch_size == 16


def test_classifier_build():
    """build() returns NeuralNetRegressor even for classification losses
    so that predict() returns probabilities, not class indices."""
    mai.init(problem_type="classification")

    settings = {
        "structural_params": {
            "Dense_hidden": {"units": 32, "activation": "relu"},
            "Dense_output": {"units": 10, "activation": "softmax"},
        },
        "optimizer": "Adam",
        "Adam": {"learning_rate": 0.001},
        "compile_params": {"loss": "categorical_crossentropy"},
        "fitting_params": {"batch_size": 32, "epochs": 5},
    }
    hypermodel = nnHyperModel(settings, input_shape=(64,), name="")
    trial = optuna.trial.FixedTrial({})
    model = hypermodel.build(trial)
    model.initialize()

    assert isinstance(model, NeuralNetRegressor)
    children = list(model.module_.net.children())
    assert len(children) == 2
    assert children[-1].linear.out_features == 10


def test_missing_epochs_raises():
    """Omitting epochs from fitting_params must raise RuntimeError."""
    mai.init(problem_type="regression")
    settings = {
        "structural_params": {"Dense_out": {"units": 1}},
        "optimizer": "Adam",
        "Adam": {"learning_rate": 0.001},
        "compile_params": {"loss": "mse"},
        "fitting_params": {"batch_size": 32},
    }
    hypermodel = nnHyperModel(settings, input_shape=(4,), name="")
    with pytest.raises(RuntimeError, match="epochs"):
        hypermodel.build(optuna.trial.FixedTrial({}))


def test_callbacks_warning():
    """Passing callbacks in fitting_params must emit a UserWarning."""
    mai.init(problem_type="regression")
    settings = {
        "structural_params": {"Dense_out": {"units": 1}},
        "optimizer": "Adam",
        "Adam": {"learning_rate": 0.001},
        "compile_params": {"loss": "mse"},
        "fitting_params": {"batch_size": 32, "epochs": 5, "callbacks": [object()]},
    }
    hypermodel = nnHyperModel(settings, input_shape=(4,), name="")
    with pytest.warns(UserWarning, match="callbacks"):
        hypermodel.build(optuna.trial.FixedTrial({}))


def test_get_search_space():
    """get_search_space() returns a dict mapping each sampled param name to
    its list of valid values, without requiring a live Optuna study."""
    mai.init(problem_type="regression")
    settings = {
        "structural_params": {
            "Dense_hidden": {
                "units": Choice([32, 64, 128]),
                "activation": "relu",
            },
            "Dense_output": {"units": 4, "activation": "linear"},
        },
        "optimizer": "Adam",
        "Adam": {"learning_rate": Choice([0.001, 0.0001])},
        "compile_params": {"loss": "mse"},
        "fitting_params": {"batch_size": 32, "epochs": 10},
    }
    hypermodel = nnHyperModel(settings, input_shape=(8,), name="")
    space = hypermodel.get_search_space()

    assert "Dense_hidden_0_units" in space
    assert space["Dense_hidden_0_units"] == [32, 64, 128]
    assert "Adam_learning_rate" in space
    assert space["Adam_learning_rate"] == [0.001, 0.0001]
