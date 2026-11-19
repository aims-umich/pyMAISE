import numpy as np
import xarray as xr
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import ShuffleSplit

import pyMAISE as mai
from pyMAISE import PostProcessor
from pyMAISE.methods.nn import MCDropout
from pyMAISE.preprocessing import one_hot_encode, train_test_split


def simulate_regression_data():
    """Simulate small random dataset for regression testing."""
    X, y = make_regression(n_samples=50, n_features=2, noise=0.1, random_state=42)
    x_raw = xr.DataArray(X)
    y_raw = xr.DataArray(y.reshape(-1, 1))
    return train_test_split([x_raw, y_raw], test_size=0.3)


def simulate_classification_data():
    """Simulate small random dataset for classification testing."""
    X, y = make_classification(
        n_samples=50, n_features=4, n_classes=2, random_state=42
    )
    x_raw = xr.DataArray(X, dims=["samples", "features"])
    y_raw = xr.DataArray(y.reshape(-1, 1), dims=["samples", "variables"]).astype(
        "object"
    )
    y_raw.coords["variables"] = ["class"]
    y_enc = one_hot_encode(y_raw)
    return train_test_split([x_raw, y_enc], test_size=0.3)


def test_mc_dropout_regression():
    """Test MCDropout end-to-end on a regression problem."""
    global_settings = mai.init(
        problem_type=mai.ProblemType.REGRESSION,
        random_state=42,
        num_configs_saved=1,
        verbosity=0,
    )

    data = simulate_regression_data()
    xtrain, xtest, ytrain, ytest = data

    parameters = {
        "models": ["MCD"],
        "MCD": {
            "num_passes": 10,
            "structural_params": {
                "Dense_1": {
                    "units": mai.Choice([16, 32]),
                    "activation": "relu",
                },
                "MCDropout_1": {
                    "rate": 0.2,
                },
                "Dense_2": {
                    "units": ytrain.shape[-1],
                    "activation": "linear",
                },
            },
            "optimizer": "Adam",
            "Adam": {
                "learning_rate": mai.Choice([1e-3, 1e-2]),
            },
            "compile_params": {
                "loss": "mean_absolute_error",
            },
            "fitting_params": {
                "epochs": 2,
                "batch_size": 16,
            },
        },
    }

    tuner = mai.Tuner(xtrain, ytrain, model_settings=parameters)
    results = tuner.nn_grid_search(
        objective="r2_score",
        cv=ShuffleSplit(
            n_splits=2, test_size=0.2, random_state=global_settings.random_state
        ),
    )

    post_processor = PostProcessor(data=data, model_configs=[results])
    metrics = post_processor.metrics()
    model = post_processor.get_model(model_type="MCD")

    assert isinstance(model, MCDropout)
    assert model.num_passes == 10
    assert metrics["Test R2"].notna().all()

    # Test predicting with uncertainty
    uncertainty_results = model.predict_with_uncertainty(xtest.values)
    assert "predictions" in uncertainty_results
    assert "mean" in uncertainty_results
    assert "epistemic_var" in uncertainty_results
    assert "aleatoric_var" in uncertainty_results

    assert uncertainty_results["predictions"].shape == (
        10,
        xtest.shape[0],
        ytrain.shape[-1],
    )
    assert uncertainty_results["mean"].shape == (xtest.shape[0], ytrain.shape[-1])
    assert uncertainty_results["epistemic_var"].shape == (
        xtest.shape[0],
        ytrain.shape[-1],
    )
    assert uncertainty_results["aleatoric_var"] is None


def test_mc_dropout_classification():
    """Test MCDropout end-to-end on a classification problem."""
    global_settings = mai.init(
        problem_type=mai.ProblemType.CLASSIFICATION,
        random_state=42,
        num_configs_saved=1,
        verbosity=0,
    )

    data = simulate_classification_data()
    xtrain, xtest, ytrain, ytest = data

    parameters = {
        "models": ["MCD"],
        "MCD": {
            "num_passes": 5,
            "structural_params": {
                "Dense_1": {
                    "units": 16,
                    "activation": "relu",
                },
                "Dropout_1": {
                    "rate": 0.2,
                },
                "Dense_2": {
                    "units": ytrain.shape[-1],
                    "activation": "softmax",
                },
            },
            "optimizer": "Adam",
            "Adam": {
                "learning_rate": 1e-3,
            },
            "compile_params": {
                "loss": "categorical_crossentropy",
            },
            "fitting_params": {
                "epochs": 2,
                "batch_size": 16,
            },
        },
    }

    tuner = mai.Tuner(xtrain, ytrain, model_settings=parameters)
    results = tuner.nn_grid_search(
        objective="accuracy",
        cv=ShuffleSplit(
            n_splits=2, test_size=0.2, random_state=global_settings.random_state
        ),
    )

    post_processor = PostProcessor(data=data, model_configs=[results])
    metrics = post_processor.metrics()
    model = post_processor.get_model(model_type="MCD")

    assert isinstance(model, MCDropout)
    assert metrics["Test Accuracy"].notna().all()

    uncertainty_results = model.predict_with_uncertainty(xtest.values)
    assert uncertainty_results["predictions"].shape == (
        5,
        xtest.shape[0],
        ytrain.shape[-1],
    )
    assert uncertainty_results["mean"].shape == (xtest.shape[0], ytrain.shape[-1])
    assert uncertainty_results["epistemic_var"].shape == (xtest.shape[0],)
    assert uncertainty_results["aleatoric_var"] is None


if __name__ == "__main__":
    test_mc_dropout_regression()
    test_mc_dropout_classification()
