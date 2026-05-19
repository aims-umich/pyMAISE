"""
binary_case_2.py.

Script for hyperparameter tuning LSTM and GRU using 2D data with rolling windows.

Note: CNN-LSTM (TimeDistributed Conv → LSTM) was removed because the
TimeDistributed wrapper pattern is not supported by the PyTorch backend.
"""

import pickle

import matplotlib.pyplot as plt
import numpy as np
import settings
from preprocessing import load_anomaly_data, split_sequences, plot_label_frequency
from sklearn.model_selection import TimeSeriesSplit

import pyMAISE as mai

print("\nBinary case 2")

# Initialize pyMAISE
global_settings = mai.init(
    problem_type=settings.problem_type,
    verbosity=settings.verbosity,
    random_state=settings.random_state,
    cuda_visible_devices="1",  # Use GPU 1
)

# Load training/testing data
data = load_anomaly_data(
    global_settings=global_settings,
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

# Plot label frequency
plot_label_frequency(ytrain, ytest, "./figs/bc2_frequency.png")

# NN structure
lstm_structure = {
    "LSTM_input": {
        "units": mai.Int(min_value=25, max_value=200),
        "return_sequences": True,
    },
    "LSTM": {
        "num_layers": mai.Int(0, 4),
        "units": mai.Int(min_value=25, max_value=200),
        "return_sequences": True,
    },
    "LSTM_output": {
        "units": mai.Int(min_value=25, max_value=200),
    },
    "Dense": {
        "num_layers": mai.Int(0, 4),
        "units": mai.Int(min_value=25, max_value=300),
        "activation": "relu",
    },
    "Dense_output": {
        "units": ytrain.shape[-1],
        "activation": "sigmoid",
    },
}

gru_structure = {
    "GRU_input": {
        "units": mai.Int(min_value=25, max_value=200),
        "return_sequences": True,
    },
    "GRU": {
        "num_layers": mai.Int(0, 4),
        "units": mai.Int(min_value=25, max_value=200),
        "return_sequences": True,
    },
    "GRU_output": {
        "units": mai.Int(min_value=25, max_value=200),
    },
    "Dense": {
        "num_layers": mai.Int(0, 4),
        "units": mai.Int(min_value=25, max_value=300),
        "activation": "relu",
    },
    "Dense_output": {
        "units": ytrain.shape[-1],
        "activation": "sigmoid",
    },
}

model_settings = {
    "models": ["LSTM", "GRU"],
    "LSTM": {
        "structural_params": lstm_structure,
        "optimizer": "Adam",
        "Adam": {
            "learning_rate": mai.Float(1e-5, 0.001),
        },
        "compile_params": {
            "loss": "categorical_crossentropy",
        },
        "fitting_params": {
            "batch_size": mai.Choice([32, 64, 128]),
            "epochs": 7,
            "validation_split": 0.10,
        },
    },
    "GRU": {
        "structural_params": gru_structure,
        "optimizer": "Adam",
        "Adam": {
            "learning_rate": mai.Float(1e-5, 0.001),
        },
        "compile_params": {
            "loss": "categorical_crossentropy",
        },
        "fitting_params": {
            "batch_size": mai.Choice([32, 64, 128]),
            "epochs": 7,
            "validation_split": 0.10,
        },
    },
}


tuner = mai.Tuner(xtrain, ytrain, model_settings=model_settings)

# Hyperparameter tuning
configs = tuner.nn_bayesian_search(
    objective="accuracy_score",
    n_trials=50,
    cv=TimeSeriesSplit(n_splits=5),
)

# Save results to pickle
with open("./configs/binary_case_2.pkl", "wb") as f:
    pickle.dump(configs, f)

# Plot convergence
plt.clf()
tuner.convergence_plot()
plt.ylim([0, 1])
plt.savefig("./figs/bc2_convergence.png", dpi=300)
