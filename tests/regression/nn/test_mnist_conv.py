import numpy as np
import pandas as pd
import xarray as xr
from sklearn.datasets import load_digits
from sklearn.model_selection import ShuffleSplit

import pyMAISE as mai
from pyMAISE.methods import nnHyperModel
from pyMAISE.preprocessing import one_hot_encode, train_test_split


def test_digits_conv():
    """End-to-end test of the Conv2D classification pipeline.

    Uses sklearn's digits dataset (1797 × 8×8 greyscale images, 10 classes)
    instead of MNIST so there is no external download dependency.  The data is
    shaped channels-first (N, 1, 8, 8) for PyTorch Conv2d, and labels are
    one-hot encoded for categorical_crossentropy.  fit() converts them to
    (N,) long integer indices internally as required by CrossEntropyLoss.
    The Dense layer after Flatten uses LazyLinear to infer its input size
    at runtime since the flat spatial size is not known at build time.
    """
    _ = mai.init(
        problem_type=mai.ProblemType.CLASSIFICATION,
        verbosity=1,
        num_configs_saved=1,
        random_state=42,
        cuda_visible_devices="-1",
    )

    # Load sklearn digits: (1797, 64) float, labels 0-9
    digits = load_digits()
    X = digits.data.astype("float32") / 16.0
    X = X.reshape(-1, 1, 8, 8)  # channels-first: (N, C, H, W)

    x = xr.DataArray(X, dims=["samples", "channels", "height", "width"])
    y = xr.DataArray(
        digits.target.reshape(-1, 1), dims=["samples", "variables"]
    ).astype("object")
    y.coords["variables"] = ["digit"]

    # One-hot encode for categorical_crossentropy
    y_enc = one_hot_encode(y)

    xtrain, xtest, ytrain, ytest = train_test_split(
        data=[x, y_enc], test_size=0.3
    )

    # Conv2D channels-first: xtrain.shape[1:] = (1, 8, 8)
    # _nn_hypermodel detects the Conv2D first layer and sets in_size = 1 (channels).
    # After Flatten, Dense uses LazyLinear to infer the flat size at runtime.
    structural = {
        "Conv2D_hidden0": {
            "filters": 16,
            "kernel_size": (3, 3),
            "activation": "relu",
            "padding": "same",
        },
        "MaxPooling2D": {"pool_size": (2, 2)},
        "Conv2D_hidden1": {
            "filters": 32,
            "kernel_size": (3, 3),
            "activation": "relu",
            "padding": "same",
        },
        "Flatten": {},
        "Dense_hidden": {
            "units": mai.Choice([64, 128]),
            "activation": "relu",
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
            "optimizer": "Adam",
            "Adam": {"learning_rate": mai.Choice([0.001, 0.0001])},
            "compile_params": {"loss": "categorical_crossentropy"},
            "fitting_params": {
                "batch_size": 32,
                "epochs": 5,
                "validation_split": 0.15,
            },
        },
    }

    tuner = mai.Tuner(xtrain, ytrain, model_settings=model_settings)

    grid_search_configs = tuner.nn_grid_search(
        objective="accuracy_score",
        cv=ShuffleSplit(n_splits=2, test_size=0.2, random_state=42),
    )

    assert isinstance(grid_search_configs["cnn"][0], pd.DataFrame)
    assert isinstance(grid_search_configs["cnn"][1], nnHyperModel)
    assert grid_search_configs["cnn"][0].shape[0] <= 1

    postprocessor = mai.PostProcessor(
        data=(xtrain, xtest, ytrain, ytest),
        model_configs=[grid_search_configs],
        new_model_settings={"cnn": {"fitting_params": {"epochs": 5}}},
    )

    metrics = postprocessor.metrics()
    assert metrics.shape[0] <= 1
    assert metrics["Train Accuracy"].notna().all()
    assert metrics["Test Accuracy"].notna().all()
