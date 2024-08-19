import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xarray as xr

# Import mnist dataset
from tensorflow.keras.datasets import mnist

import pyMAISE as mai
from pyMAISE.methods import nnHyperModel
from pyMAISE.preprocessing import one_hot_encode, train_test_split


@pytest.fixture
def mnist_data():
    # Load dataset
    (xtrain, ytrain), (xtest, ytest) = mnist.load_data()

    # Reshape dataset to have a single channel
    xtrain = xtrain[..., np.newaxis].astype("float32") / 255.0
    xtest = xtest[..., np.newaxis].astype("float32") / 255.0

    ytrain = ytrain.reshape((ytrain.shape[0], 1))
    ytest = ytest.reshape((ytest.shape[0], 1))

    # Combine data so preprocessor can do train test split
    x = np.concatenate((xtrain, xtest))
    y = np.concatenate((ytrain, ytest))

    # Create xarrays
    x = xr.DataArray(x, dims=["samples", "height", "width", "variables"])
    y = xr.DataArray(y, dims=["samples", "variables"]).astype("object")
    y.coords["variables"] = ["number"]

    return x, y


def test_mnist_conv(mnist_data):
    # Initialize pyMAISE
    _ = mai.init(
        problem_type=mai.ProblemType.CLASSIFICATION,
        verbosity=1,
        num_configs_saved=1,
        random_state=42,
        cuda_visible_devices="-1",  # Use CPUs only
    )

    # Give data to preprocessor and do train/test split
    inputs, outputs = mnist_data

    # One hot encode multiclass outputs
    outputs = one_hot_encode(outputs)

    # Split data
    xtrain, xtest, ytrain, ytest = train_test_split(
        data=[inputs, outputs], test_size=0.3
    )

    # Using Sve Final Model archetecture
    structural = {
        "Conv2D_hidden0": {
            "filters": 32,
            "kernel_size": (3, 3),
            "activation": "relu",
            "kernel_initializer": "he_uniform",
            "input_shape": (28, 28, 1),
        },
        "MaxPooling2D": {
            "pool_size": (2, 2),
        },
        "Conv2D_hidden1": {
            "filters": 32,
            "kernel_size": (3, 3),
            "activation": "relu",
            "kernel_initializer": "he_uniform",
            "input_shape": (28, 28, 1),
        },
        "Conv2D_hidden2": {
            "filters": 32,
            "kernel_size": (3, 3),
            "activation": "relu",
            "kernel_initializer": "he_uniform",
            "input_shape": (28, 28, 1),
        },
        "MaxPooling2D_hidden1": {
            "pool_size": (2, 2),
        },
        "Flatten": {},
        "Dense_hidden": {
            "units": 100,
            "activation": "relu",
            "kernel_initializer": "he_uniform",
        },
        "Dense_output": {
            "units": 10,
            "activation": "softmax",
        },
    }

    model_settings = {
        "models": ["cnn"],
        "cnn": {
            "structural_params": structural,
            "optimizer": "SGD",
            "SGD": {
                "learning_rate": mai.Choice([0.0001, 0.01]),
                "momentum": 0.9,
            },
            "compile_params": {
                "loss": "categorical_crossentropy",
                "metrics": ["accuracy"],
            },
            "fitting_params": {
                "batch_size": 32,
                "epochs": 10,
                "validation_split": 0.15,
            },
        },
    }

    tuner = mai.Tuner(xtrain, ytrain, model_settings=model_settings)

    # Grid search
    grid_search_configs = tuner.nn_grid_search(
        objective="accuracy_score", cv=2, max_consecutive_failed_trials=1
    )

    # Check contents of grid search results
    assert isinstance(grid_search_configs["cnn"][0], pd.DataFrame)
    assert isinstance(grid_search_configs["cnn"][1], nnHyperModel)
    assert grid_search_configs["cnn"][0].shape == (1, 1)
    assert tuner.cv_performance_data["cnn"].shape == (2, 2)

    # Model post-processing
    new_model_settings = {
        "cnn": {
            "fitting_params": {
                "epochs": 10,
            },
        },
    }
    postprocessor = mai.PostProcessor(
        data=(xtrain, xtest, ytrain, ytest),
        model_configs=[grid_search_configs],
        new_model_settings=new_model_settings,
    )

    metrics = postprocessor.metrics()
    print(metrics)

    # Shape assertion on metrics object
    assert metrics.shape == (1, 10)
    assert np.abs(metrics["Train Accuracy"][0] - 1) < 0.025
    assert np.abs(metrics["Test Accuracy"][0] - 1) < 0.025

    # Run correlation matrix
    _, axs = plt.subplots(1, 2)
    postprocessor.confusion_matrix(axs=axs)
