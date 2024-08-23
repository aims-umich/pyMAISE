import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler

import pyMAISE as mai
from pyMAISE.datasets import load_MITR
from pyMAISE.preprocessing import scale_data, train_test_split


def test_mitr():
    # ===========================================================================
    # Regression test parameters
    # Data set parameters
    num_observations = 1000
    num_features = 6
    num_outputs = 22

    # Expected performance metrics
    expected_metrics = pd.read_csv(
        os.path.dirname(__file__) + "/supporting/mitr_testing_metrics.csv"
    )

    # ===========================================================================
    # pyMAISE initialization
    global_settings = mai.init(
        problem_type=mai.ProblemType.REGRESSION,
        verbosity=1,
        random_state=42,
        num_configs_saved=1,
        cuda_visible_devices="-1",
    )

    # Assertions for global settings
    assert global_settings.verbosity == 1
    assert global_settings.random_state == 42
    assert global_settings.num_configs_saved == 1

    # Get MITR data
    data, inputs, outputs = load_MITR()

    # Assert inputs and outputs are the correct size
    assert inputs.shape[0] == num_observations and inputs.shape[1] == num_features
    assert outputs.shape[0] == num_observations and outputs.shape[1] == num_outputs

    # Train test split
    xtrain, xtest, ytrain, ytest = train_test_split(
        data=[inputs, outputs], test_size=0.3
    )
    xtrain, xtest, _ = scale_data(xtrain, xtest, MinMaxScaler())
    ytrain, ytest, _ = scale_data(ytrain, ytest, MinMaxScaler())
    data = (xtrain, xtest, ytrain, ytest)

    # Train-test split size assertions
    assert (
        data[0].shape[0] == num_observations * (1 - 0.3)
        and data[0].shape[1] == num_features
    )
    assert (
        data[1].shape[0] == num_observations * 0.3 and data[1].shape[1] == num_features
    )
    assert (
        data[2].shape[0] == num_observations * (1 - 0.3)
        and data[2].shape[1] == num_outputs
    )
    assert (
        data[3].shape[0] == num_observations * 0.3 and data[3].shape[1] == num_outputs
    )

    # ===========================================================================
    # Model initialization
    model_settings = {
        "models": ["Linear", "Lasso", "DT", "KN", "RF"],
    }
    tuning = mai.Tuner(xtrain, ytrain, model_settings=model_settings)

    # ===========================================================================
    # Hyper-parameter tuning
    grid_search_spaces = {
        "Linear": {"fit_intercept": [True, False]},
        "Lasso": {"alpha": np.linspace(0.000001, 1, 200)},
        "DT": {
            "max_depth": [None, 5, 10, 25, 50],
            "max_features": [None, "sqrt", "log2", 0.2, 0.4, 0.6, 0.8, 1],
            "min_samples_leaf": [1, 2, 4, 6, 8, 10],
            "min_samples_split": [2, 4, 6, 8, 10],
        },
        "RF": {
            "n_estimators": [50, 100, 150],
            "criterion": ["squared_error", "absolute_error"],
            "min_samples_split": [2, 4],
            "max_features": [None, "sqrt", "log2", 1],
        },
        "KN": {
            "n_neighbors": [1, 2, 4, 6, 8, 10, 14, 17, 20],
            "weights": ["uniform", "distance"],
            "leaf_size": [1, 5, 10, 15, 20, 25, 30],
        },
    }

    grid_search_configs = tuning.grid_search(
        param_spaces=grid_search_spaces,
        models=grid_search_spaces.keys(),
        cv=ShuffleSplit(
            n_splits=1, test_size=0.15, random_state=global_settings.random_state
        ),
    )

    # ===========================================================================
    # Model post-processing
    postprocessor = mai.PostProcessor(
        data=data,
        model_configs=[grid_search_configs],
    )
    metrics = postprocessor.metrics(metrics={"MSE": mean_squared_error})[
        [
            "Model Types",
            "Train MAE",
            "Train MSE",
            "Train RMSE",
            "Train R2",
            "Test MAE",
            "Test MSE",
            "Test RMSE",
            "Test R2",
        ]
    ]

    # Assert expected dataframe and results match
    print(
        "Expected Values\n",
        expected_metrics.sort_values(by=["Test R2"], ascending=False),
    )
    print("pyMAISE Values\n", metrics)
    pd.testing.assert_frame_equal(
        expected_metrics.sort_values(by=["Test R2"], ascending=False), metrics
    )
