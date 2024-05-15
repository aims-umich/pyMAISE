"""
binary_case_2.py.

inputs  (samples * timesteps, features)
outputs (samples * timesteps, features)

TODO:
- Save convergence plot to ./figs/
- Add further hyperparameter tuning to LSTM
- Add GRU
- Add CNN-LSTM

See how non_faulty_frac and timestep_step change the results.
Increase number of trials and cross-validation splits once
you're feeling good about everything.
"""

import pickle
from sklearn.model_selection import TimeSeriesSplit

import pyMAISE as mai
import settings
from preprocessing import load_anomaly_data, split_sequences

print("\nBinary case 2")

# Initialize pyMAISE
mai.init(
    problem_type=settings.problem_type,
    verbosity=settings.verbosity,
    random_state=settings.random_state,
    cuda_visible_devices="1",  # Use GPU 1
)

# Load training/testing data
data = load_anomaly_data(
    stack_series=True,
    multiclass=False,
    test_size=settings.test_size,
    non_faulty_frac=settings.non_faulty_frac,
    timestep_step=10,
)

# Combine data and create rolling windows using SplitSequence
xtrain, xtest, ytrain, ytest = split_sequences(
    data=data[:-1],
    input_steps=10,
    output_steps=1,
    output_position=0,
)

# Model initialization
lstm_structure = {
    "LSTM_input": {
        "units": 50,
        "input_shape": xtrain.shape[1:],
        "activation": "tanh",
        "recurrent_activation": "sigmoid",
        "return_sequences": True,
    },
    "LSTM": {
        "num_layers": mai.Int(0, 3),
        "units": 50,
        "activation": "tanh",
        "recurrent_activation": "sigmoid",
        "return_sequences": True,
    },
    "LSTM_output": {
        "units": 50,
        "activation": "tanh",
        "recurrent_activation": "sigmoid",
    },
    "Dense": {
        "num_layers": mai.Int(0, 3),
        "units": 25,
        "activation": "relu",
    },
    "Dense_output": {
        "units": ytrain.shape[-1],
        "activation": "sigmoid",
    },
}

model_settings = {
    "models": ["LSTM"],
    "LSTM": {
        "structural_params": lstm_structure,
        "optimizer": "Adam",
        "Adam": {"learning_rate": mai.Float(1e-5, 0.001)},
        "compile_params": {
            "loss": "binary_crossentropy",
            "metrics": ["accuracy"],
        },
        "fitting_params": {
            "batch_size": mai.Choice([64, 128, 256]),
            "epochs": 5,
            "validation_split": 0.15,
        },
    },
}

tuner = mai.Tuner(xtrain, ytrain, model_settings=model_settings)

# Hyperparameter tuning
configs = tuner.nn_bayesian_search(
    objective="accuracy_score",
    max_trials=2,
    cv=TimeSeriesSplit(n_splits=2),
)

# Save results to pickle
with open("./configs/binary_case_2.pkl", "wb") as f:
    pickle.dump(configs, f)
