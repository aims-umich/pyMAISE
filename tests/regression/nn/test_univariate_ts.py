import pandas as pd
import pytest
import xarray as xr
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler

import pyMAISE as mai
from pyMAISE.methods import nnHyperModel
from pyMAISE.preprocessing import SplitSequence, scale_data, train_test_split


def test_nn_lstm_univariate_series():
    # Link to tutorial: https://machinelearningmastery.com/time-series-
    # prediction-lstm-recurrent-neural-networks-python-keras/

    # Initialize pyMAISE
    global_settings = mai.init(
        problem_type=mai.ProblemType.REGRESSION,
        verbosity=1,
        random_state=42,
        num_configs_saved=1,
        cuda_visible_devices="-1",
    )

    # Get univariate timeseries data (airline data)
    data = pd.read_csv(
        "https://raw.githubusercontent.com/jbrownlee/"
        + "Datasets/master/airline-passengers.csv"
    )

    # Define raw sequence data and pass to preprocessor
    data = xr.DataArray(
        data.iloc[:, 1].values.reshape(144, 1),
        coords={"timesteps": data.iloc[:, 0].values, "features": ["passengers"]},
    )

    # Assert data shapes
    assert data.shape == (144, 1)

    # Split data and confirm shapes and contents of first 2 samples
    split_sequence = SplitSequence(
        input_steps=1,
        output_steps=1,
        output_position=1,
        sequence_inputs=["passengers"],
        sequence_outputs=["passengers"],
    )
    inputs, outputs = split_sequence.split(data)
    assert inputs.shape == (
        143,
        1,
        1,
    )
    assert outputs.shape == (
        143,
        1,
    )

    # Train test split and scaling
    xtrain, xtest, ytrain, ytest = train_test_split(
        data=[inputs, outputs], test_size=0.3
    )
    xtrain, xtest, _ = scale_data(xtrain, xtest, MinMaxScaler())
    ytrain, ytest, yscaler = scale_data(ytrain, ytest, MinMaxScaler())

    # Testing if split is is 20 %
    assert xtrain.shape == (100, 1, 1)
    assert ytrain.shape == (100, 1)
    assert xtest.shape == (43, 1, 1)
    assert ytest.shape == (43, 1)

    # Creating RNN model
    structural_hyperparameters = {
        "LSTM_hidden": {
            "units": 4,
        },
        "Dense_output": {
            "units": 1,
        },
    }
    model_settings = {
        "models": ["rnn"],
        "rnn": {
            "structural_params": structural_hyperparameters,
            "optimizer": "Adam",
            "Adam": {
                "learning_rate": 0.001,
            },
            "compile_params": {
                "loss": "mean_squared_error",
                "metrics": ["mean_squared_error"],
            },
            "fitting_params": {
                "batch_size": 1,
                "epochs": 100,
            },
        },
    }

    # Passing preprocessed data into tuner function for HP tuning
    tuner = mai.Tuner(xtrain, ytrain, model_settings=model_settings)

    # Grid search
    grid_search_configs = tuner.nn_grid_search(
        objective="mean_squared_error",
        cv=ShuffleSplit(n_splits=1, random_state=global_settings.random_state),
    )
    assert isinstance(grid_search_configs["rnn"][0], pd.DataFrame)
    assert isinstance(grid_search_configs["rnn"][1], nnHyperModel)
    assert grid_search_configs["rnn"][0].shape == (1, 1)
    assert tuner.cv_performance_data["rnn"].shape == (2, 1)

    # Model post-processing
    new_model_settings = {
        "rnn": {
            "fitting_params": {
                "epochs": 100,
            },
        },
    }
    postprocessor = mai.PostProcessor(
        data=(xtrain, xtest, ytrain, ytest),
        model_configs=[grid_search_configs],
        new_model_settings=new_model_settings,
        yscaler=yscaler,
    )
    print(postprocessor.metrics()[["Train RMSE", "Test RMSE"]].values.tolist()[0])
    # Tutorial
    # Train 22.68 RMSE
    # Test Score: 49.34 RMSE
    # Compare performance metrics
    # Do to the slight differences in train/test split we have different results
    assert postprocessor.metrics()[["Train RMSE", "Test RMSE"]].values.tolist()[
        0
    ] == pytest.approx([22.68, 49.34], 13)
