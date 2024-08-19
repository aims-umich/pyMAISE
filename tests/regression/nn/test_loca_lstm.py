import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler

import pyMAISE as mai
from pyMAISE.datasets import load_loca
from pyMAISE.methods import nnHyperModel
from pyMAISE.preprocessing import SplitSequence, scale_data, train_test_split


def test_loca_lstm():
    # Initialize pyMAISE
    _ = mai.init(
        problem_type=mai.ProblemType.REGRESSION,
        verbosity=1,
        num_configs_saved=2,
        random_state=42,
        cuda_visible_devices="-1",  # Use CPUs only
    )

    # Load LOCA data and shrink dataset for test speed
    _, perturbed = load_loca(stack_series=False)
    perturbed = perturbed[:1000, :, :]

    # Split sequence data
    split_sequences = SplitSequence(
        input_steps=4,
        output_steps=1,
        output_position=1,
        sequence_inputs=range(-4, 0),
        sequence_outputs=range(-4, 0),
        const_inputs=range(40),
    )
    inputs, outputs = split_sequences.split(perturbed)
    assert perturbed.shape == (1000, 400, 44)
    assert inputs.shape == (1000, 396, 56)
    assert outputs.shape == (1000, 396, 4)

    # Train test split data
    xtrain, xtest, ytrain, ytest = train_test_split(
        data=[inputs, outputs], test_size=0.3
    )
    xtrain, xtest, _ = scale_data(xtrain, xtest, MinMaxScaler())
    ytrain, ytest, yscaler = scale_data(ytrain, ytest, MinMaxScaler())
    split_data = (xtrain, xtest, ytrain, ytest)

    assert split_data[0].shape == (700, 396, 56)
    assert split_data[1].shape == (300, 396, 56)
    assert split_data[2].shape == (700, 396, 4)
    assert split_data[3].shape == (300, 396, 4)
    print(inputs.shape)

    # RNN model settings
    structural = {
        # "LSTM_input": {
        #     "units": 100,
        #     "activation": "tanh",
        #     "recurrent_activation": "sigmoid",
        #     "return_sequences": True,
        #     "input_shape": (396, 56),
        # },
        "LSTM_hidden0": {
            "units": 80,
            "activation": "tanh",
            "recurrent_activation": "sigmoid",
            "return_sequences": True,
        },
        "LSTM_hidden1": {
            "units": 60,
            "activation": "tanh",
            "recurrent_activation": "sigmoid",
            "return_sequences": True,
        },
        "LSTM_hidden2": {
            "units": 40,
            "activation": "tanh",
            "recurrent_activation": "sigmoid",
            "return_sequences": True,
        },
        "Dense_output": {
            "units": 4,
            "activation": "linear",
        },
    }
    model_settings = {
        "models": ["rnn"],
        "rnn": {
            "structural_params": structural,
            "optimizer": "Adam",
            "Adam": {
                "learning_rate": mai.Choice([0.0001, 0.001]),
                "clipnorm": 1.0,
                "clipvalue": 0.5,
            },
            "compile_params": {
                "loss": "mean_absolute_error",
                "metrics": ["mean_absolute_error"],
            },
            "fitting_params": {"batch_size": 16, "epochs": 5, "validation_split": 0.15},
        },
    }
    tuner = mai.Tuner(xtrain, ytrain, model_settings=model_settings)

    # Grid search
    grid_search_configs = tuner.nn_grid_search(
        objective="r2_score",
        cv=TimeSeriesSplit(n_splits=2),
    )
    assert isinstance(grid_search_configs["rnn"][0], pd.DataFrame)
    assert isinstance(grid_search_configs["rnn"][1], nnHyperModel)
    assert grid_search_configs["rnn"][0].shape == (2, 1)
    assert tuner.cv_performance_data["rnn"].shape == (2, 2)

    # Model post-processing
    new_model_settings = {
        "rnn": {
            "fitting_params": {
                "epochs": 10,
            },
        },
    }
    postprocessor = mai.PostProcessor(
        data=split_data,
        model_configs=[grid_search_configs],
        new_model_settings=new_model_settings,
        yscaler=yscaler,
    )
    assert postprocessor.metrics().shape == (2, 12)
