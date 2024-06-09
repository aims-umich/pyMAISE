"""
binary_case_2.py.

Script for hyperparameter tuning LSTM and GRU using 3D data.
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

# Plot label frequency
plot_label_frequency(ytrain, ytest, "./figs/bc2_frequency.png")

# NN structure
lstm_structure = {
    "LSTM_input": {
        "units": mai.Int(min_value=25, max_value=150),
        "input_shape": xtrain.shape[1:],
        "activation": "tanh",
        "recurrent_activation": "sigmoid",
        "return_sequences": True,
    },
    "LSTM": {
        "num_layers": mai.Int(0, 4),
        "units": mai.Int(min_value=25, max_value=150),
        "activation": mai.Choice(["tanh", "sigmoid"]),
        "recurrent_activation": "sigmoid",
        "return_sequences": True,
    },
    "LSTM_output": {
        "units": mai.Int(min_value=25, max_value=150),
        "activation": mai.Choice(["tanh", "sigmoid"]),
        "recurrent_activation": "sigmoid",
    },
    "Dense": {
        "num_layers": mai.Int(0, 4),
        "units": mai.Int(min_value=25, max_value=250),
        "activation": "relu",
    },
    "Dense_output": {
        "units": ytrain.shape[-1],
        "activation": "sigmoid",
    },
}

gru_structure = {
    "GRU_input": {
        "units": mai.Int(min_value=25, max_value=150),
        "input_shape": xtrain.shape[1:],
        "activation": "tanh",
        "recurrent_activation": "sigmoid",
        "return_sequences": True,
    },
    "GRU": {
        "num_layers": mai.Int(0, 4),
        "units": mai.Int(min_value=25, max_value=150),
        "activation": mai.Choice(["tanh", "sigmoid"]),
        "recurrent_activation": "sigmoid",
        "return_sequences": True,
    },
    "GRU_output": {
        "units": mai.Int(min_value=25, max_value=150),
        "activation": mai.Choice(["tanh", "sigmoid"]),
        "recurrent_activation": "sigmoid",
    },
    "Dense": {
        "num_layers": mai.Int(0, 4),
        "units": mai.Int(min_value=25, max_value=250),
        "activation": "relu",
    },
    "Dense_output": {
        "units": ytrain.shape[-1],
        "activation": "sigmoid",
    },
}

cnn_lstm_structure = {
    "Conv1D_input": {
        "filters": mai.Int(min_value=32, max_value=128),
        "kernel_size": mai.Int(min_value=2, max_value=5),
        "activation": "relu",
        "input_shape": xtrain.shape[1:],
    },
    "Conv1D": {
        "num_layers": mai.Int(0, 4),
        "filters": mai.Int(min_value=32, max_value=128),
        "kernel_size": mai.Int(min_value=2, max_value=5),
        "activation": "relu",
    },
    "LSTM": {
        "num_layers": mai.Int(1, 4),
        "units": mai.Int(min_value=25, max_value=150),
        "activation": mai.Choice(["tanh", "sigmoid"]),
        "recurrent_activation": "sigmoid",
        "return_sequences": True,
    },
    "LSTM_output": {
        "units": mai.Int(min_value=25, max_value=150),
        "activation": mai.Choice(["tanh", "sigmoid"]),
        "recurrent_activation": "sigmoid",
    },
    "Dense": {
        "num_layers": mai.Int(0, 4),
        "units": mai.Int(min_value=25, max_value=250),
        "activation": "relu",
    },
    "Dense_output": {
        "units": ytrain.shape[-1],
        "activation": "sigmoid",
    },
}

model_settings = {
    "models": ["CNN-LSTM"],
    "LSTM": {
        "structural_params": lstm_structure,
        "optimizer": "Adam",
        "Adam": {
            "learning_rate": mai.Float(1e-5, 0.001),
            "clipnorm": mai.Float(0.8, 1.2),
            "clipvalue": mai.Float(0.3, 0.7),
        },
        "compile_params": {
            "loss": "binary_crossentropy",
            "metrics": ["accuracy"],
        },
        "fitting_params": {
            "batch_size": mai.Choice([32, 64, 128]),
            "epochs": 5,
            "validation_split": 0.15,
        },
    },
    "GRU": {
        "structural_params": gru_structure,
        "optimizer": "Adam",
        "Adam": {
            "learning_rate": mai.Float(1e-5, 0.001),
            "clipnorm": mai.Float(0.8, 1.2),
            "clipvalue": mai.Float(0.3, 0.7),
        },
        "compile_params": {
            "loss": "binary_crossentropy",
            "metrics": ["accuracy"],
        },
        "fitting_params": {
            "batch_size": mai.Choice([32, 64, 128]),
            "epochs": 5,
            "validation_split": 0.15,
        },
    },
    "CNN-LSTM": {
        "structural_params": cnn_lstm_structure,
        "optimizer": "Adam",
        "Adam": {
            "learning_rate": mai.Float(1e-5, 0.001),
            "clipnorm": mai.Float(0.8, 1.2),
            "clipvalue": mai.Float(0.3, 0.7),
        },
        "compile_params": {
            "loss": "binary_crossentropy",
            "metrics": ["accuracy"],
        },
        "fitting_params": {
            "batch_size": mai.Choice([32, 64, 128]),
            "epochs": 5,
            "validation_split": 0.15,
        },
    },
}


tuner = mai.Tuner(xtrain, ytrain, model_settings=model_settings)

# Hyperparameter tuning
configs = tuner.nn_bayesian_search(
    objective="accuracy_score",
    max_trials=1,
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