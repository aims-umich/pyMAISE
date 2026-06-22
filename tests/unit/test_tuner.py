import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from skopt.space import Categorical, Real

import pyMAISE as mai
from pyMAISE.datasets import load_fp, load_xs
from pyMAISE.methods import nnHyperModel
from pyMAISE.preprocessing import scale_data, train_test_split


# ================================================================
@pytest.fixture
def setup_xr():
    # Initialize pyMAISE
    _ = mai.init(
        problem_type=mai.ProblemType.REGRESSION,
        verbosity=2,
        random_state=42,
        num_configs_saved=2,
        new_nn_architecture=True,
    )

    # Load data
    _, inputs, outputs = load_xs()

    # Train/test split data
    xtrain, xtest, ytrain, ytest = train_test_split(
        data=[inputs, outputs], test_size=0.3
    )

    # Scale data
    xtrain, xtest, _ = scale_data(xtrain, xtest, scaler=MinMaxScaler())
    ytrain, ytest, _ = scale_data(ytrain, ytest, scaler=MinMaxScaler())

    # Load reactor physics PreProcessor data and scale
    return xtrain, xtest, ytrain, ytest


@pytest.fixture
def setup_fp():
    # Initialize pyMAISE
    _ = mai.init(
        problem_type=mai.ProblemType.REGRESSION,
        verbosity=2,
        random_state=42,
        num_configs_saved=2,
        new_nn_architecture=True,
    )

    # Load data
    _, inputs, outputs = load_fp()

    # Train/test split data
    xtrain, xtest, ytrain, ytest = train_test_split(
        data=[inputs, outputs], test_size=0.3
    )

    # Scale data
    xtrain, xtest, _ = scale_data(xtrain, xtest, scaler=MinMaxScaler())
    ytrain, ytest, _ = scale_data(ytrain, ytest, scaler=MinMaxScaler())

    # Load reactor physics PreProcessor data and scale
    return xtrain, xtest, ytrain, ytest


@pytest.fixture()
def setup_nn_model_settings(setup_xr, setup_fp):
    data = []
    for xtrain, xtest, ytrain, ytest in [setup_xr, setup_fp]:
        # FNN structural parameters
        structural = {
            "Dense_input": {
                "units": mai.Choice([100, 200, 300]),
                "input_dim": xtrain.shape[-1],
                "activation": "relu",
                "kernel_initializer": "normal",
                "sublayer": "Dropout",
                "Dropout": {"rate": 0.5},
            },
            "Dense_output": {
                "units": ytrain.shape[-1],
                "activation": "linear",
                "kernel_initializer": "normal",
            },
        }

        # Training settings
        model_settings = {
            "models": ["fnn"],
            "fnn": {
                "structural_params": structural,
                "optimizer": "Adam",
                "Adam": {
                    "learning_rate": 0.001,
                },
                "compile_params": {
                    "loss": "mean_absolute_error",
                    "metrics": ["mean_absolute_error"],
                },
                "fitting_params": {
                    "batch_size": 8,
                    "epochs": 10,
                    "validation_split": 0.15,
                },
            },
        }
        data.append(([xtrain, xtest, ytrain, ytest], model_settings))
    return data


# ================================================================
def test_grid_search(setup_xr, setup_fp):
    for xtrain, _, ytrain, _ in [setup_xr, setup_fp]:
        # Build Tuner object
        model_settings = {
            "models": ["Linear", "Lasso"],
        }
        tuner = mai.Tuner(xtrain, ytrain, model_settings=model_settings)

        # Run grid search over search spaces
        spaces = {
            "Linear": {"fit_intercept": [True, False]},
            "Lasso": {"alpha": np.linspace(0.000001, 1, 5)},
        }
        search_data = tuner.grid_search(
            param_spaces=spaces,
            cv=ShuffleSplit(n_splits=1, test_size=0.15, random_state=42),
        )

        # Assert that we have the model wrappers
        assert isinstance(search_data["Linear"][0], pd.DataFrame)
        assert isinstance(search_data["Lasso"][0], pd.DataFrame)
        assert isinstance(search_data["Linear"][1], LinearRegression)
        assert isinstance(search_data["Lasso"][1], Lasso)

        # Assert search data dimensions
        assert search_data.keys() == spaces.keys()
        assert search_data["Linear"][0].shape == (2, 1)
        assert search_data["Lasso"][0].shape == (2, 1)
        assert tuner.cv_performance_data["Linear"].shape == (2, 2)
        assert tuner.cv_performance_data["Lasso"].shape == (2, 5)


# ================================================================
def test_random_search(setup_xr, setup_fp):
    for xtrain, _, ytrain, _ in [setup_xr, setup_fp]:
        # Build Tuner object
        model_settings = {
            "models": ["Linear", "Lasso"],
        }
        tuner = mai.Tuner(xtrain, ytrain, model_settings=model_settings)

        # Run random search over search spaces
        spaces = {
            "Linear": {"fit_intercept": [True, False]},
            "Lasso": {"alpha": np.linspace(0.000001, 1, 5)},
        }
        search_data = tuner.random_search(
            param_spaces=spaces,
            n_iter=5,
            cv=ShuffleSplit(n_splits=1, test_size=0.15, random_state=42),
        )

        # Assert that we have the model wrappers
        assert isinstance(search_data["Linear"][0], pd.DataFrame)
        assert isinstance(search_data["Lasso"][0], pd.DataFrame)
        assert isinstance(search_data["Linear"][1], LinearRegression)
        assert isinstance(search_data["Lasso"][1], Lasso)

        # Assert search data dimensions
        assert search_data.keys() == spaces.keys()
        assert search_data["Linear"][0].shape == (2, 1)
        assert search_data["Lasso"][0].shape == (2, 1)
        assert tuner.cv_performance_data["Linear"].shape == (
            2,
            2,
        )  # n_iter > space size so restrict to space size
        assert tuner.cv_performance_data["Lasso"].shape == (2, 5)


# ================================================================
def test_bayesian_search(setup_xr, setup_fp):
    for xtrain, _, ytrain, _ in [setup_xr, setup_fp]:
        # Build Tuner object
        model_settings = {
            "models": ["Linear", "Lasso"],
        }
        tuner = mai.Tuner(xtrain, ytrain, model_settings=model_settings)

        # Run bayesian search over search spaces
        spaces = {
            "Linear": {"fit_intercept": Categorical([True, False])},
            "Lasso": {"alpha": Real(0.000001, 1)},
        }
        search_data = tuner.bayesian_search(
            param_spaces=spaces,
            n_iter=5,
            cv=ShuffleSplit(n_splits=1, test_size=0.15, random_state=42),
        )

        # Assert that we have the model wrappers
        assert isinstance(search_data["Linear"][0], pd.DataFrame)
        assert isinstance(search_data["Lasso"][0], pd.DataFrame)
        assert isinstance(search_data["Linear"][1], LinearRegression)
        assert isinstance(search_data["Lasso"][1], Lasso)

        # Assert search data dimensions
        assert search_data.keys() == spaces.keys()
        assert search_data["Linear"][0].shape == (2, 1)
        assert search_data["Lasso"][0].shape == (2, 1)
        assert tuner.cv_performance_data["Linear"].shape == (2, 5)
        assert tuner.cv_performance_data["Lasso"].shape == (2, 5)


# ================================================================
def test_nn_grid_search(setup_nn_model_settings):
    for data, model_settings in setup_nn_model_settings:
        # Build Tuner
        tuner = mai.Tuner(data[0], data[2], model_settings=model_settings)

        # Run grid search
        search_data = tuner.nn_grid_search(
            objective="r2_score",
            cv=ShuffleSplit(n_splits=1, test_size=0.15, random_state=42),
        )

        # Assert that we have the correct types
        assert isinstance(search_data["fnn"][0], pd.DataFrame)
        assert isinstance(search_data["fnn"][1], nnHyperModel)

        # Assert search data dimensions
        assert list(search_data.keys()) == model_settings["models"]
        assert search_data["fnn"][0].shape == (2, 1)
        assert tuner.cv_performance_data["fnn"].shape == (2, 3)


# ================================================================
def test_nn_random_search(setup_nn_model_settings):
    for data, model_settings in setup_nn_model_settings:
        # Build Tuner
        tuner = mai.Tuner(data[0], data[2], model_settings=model_settings)

        # Run random search
        search_data = tuner.nn_random_search(
            objective="r2_score",
            n_trials=3,
            cv=ShuffleSplit(n_splits=1, test_size=0.15, random_state=42),
        )

        # Assert that we have the correct types
        assert isinstance(search_data["fnn"][0], pd.DataFrame)
        assert isinstance(search_data["fnn"][1], nnHyperModel)

        # Assert search data dimensions
        assert list(search_data.keys()) == model_settings["models"]
        assert search_data["fnn"][0].shape == (2, 1)
        assert tuner.cv_performance_data["fnn"].shape == (2, 3)


# ================================================================
def test_nn_bayesian_search(setup_nn_model_settings):
    for data, model_settings in setup_nn_model_settings:
        # Build Tuner
        tuner = mai.Tuner(data[0], data[2], model_settings=model_settings)

        # Run bayesian search
        search_data = tuner.nn_bayesian_search(
            objective="r2_score",
            n_trials=3,
            cv=ShuffleSplit(n_splits=1, test_size=0.15, random_state=42),
        )

        # Assert that we have the correct types
        assert isinstance(search_data["fnn"][0], pd.DataFrame)
        assert isinstance(search_data["fnn"][1], nnHyperModel)

        # Assert search data dimensions
        assert list(search_data.keys()) == model_settings["models"]
        assert search_data["fnn"][0].shape == (2, 1)
        assert tuner.cv_performance_data["fnn"].shape == (2, 3)


# ================================================================
def test_nn_hyperband_search(setup_nn_model_settings):
    for data, model_settings in setup_nn_model_settings:
        # Build Tuner
        tuner = mai.Tuner(data[0], data[2], model_settings=model_settings)

        # Run hyperband search
        search_data = tuner.nn_hyperband_search(
            objective="r2_score",
            n_trials=3,
            cv=ShuffleSplit(n_splits=1, test_size=0.15, random_state=42),
        )

        # Assert that we have the correct types
        assert isinstance(search_data["fnn"][0], pd.DataFrame)
        assert isinstance(search_data["fnn"][1], nnHyperModel)

        # Assert search data dimensions
        assert list(search_data.keys()) == model_settings["models"]
        assert search_data["fnn"][0].shape == (2, 1)
        assert tuner.cv_performance_data["fnn"].shape == (2, 3)


# ================================================================
def test_convergence_plot(setup_xr):
    xtrain, _, ytrain, _ = setup_xr

    # Build Tuner object
    model_settings = {
        "models": ["Linear", "Lasso"],
    }
    tuner = mai.Tuner(xtrain, ytrain, model_settings=model_settings)

    # Run grid search over search spaces
    spaces = {
        "Linear": {"fit_intercept": [True, False]},
        "Lasso": {"alpha": np.linspace(0.000001, 1, 5)},
    }
    _ = tuner.grid_search(
        param_spaces=spaces,
        cv=ShuffleSplit(n_splits=1, test_size=0.15, random_state=42),
    )

    # Plot convergence of linear
    _ = tuner.convergence_plot(model_types="Linear")
