"""
binary_case_1.py.

Script for hyperparameter tuning LSTM and GRU using 3D data.
"""

import pickle
from sklearn.model_selection import TimeSeriesSplit

import numpy as np
import matplotlib.pyplot as plt
import pyMAISE as mai
import settings
from preprocessing import load_anomaly_data

print("\nBinary case 1")

# Initialize pyMAISE
global_settings = mai.init(
    problem_type=settings.problem_type,
    verbosity=settings.verbosity,
    random_state=settings.random_state,
    cuda_visible_devices="1",  # Use GPU 1
)

# Load training/testing data
xtrain, xtest, ytrain, ytest, _ = load_anomaly_data(
    global_settings=global_settings,
    stack_series=False,
    multiclass=False,
    test_size=settings.test_size,
    non_faulty_frac=settings.non_faulty_frac,
    timestep_step=1,
)

# Plot label frequency
plt.clf()
plt.bar(["Fault", "Run"], np.sum(np.concatenate([ytrain, ytest], axis=0), axis=0))
plt.title("Frequency of One-Hot Fault/Non Fault")
plt.xlabel("Categories")
plt.ylabel("Frequency")
plt.ticklabel_format(style="plain", axis="y")
plt.savefig("./figs/bc1_frequency.png", dpi=300)

# Model initialization
lstm_structure = {
    "LSTM_input": {
        "units": mai.Int(min_value=25, max_value=200),
        "input_shape": xtrain.shape[1:],
        "activation": "tanh",
        "recurrent_activation": "sigmoid",
        "return_sequences": True,
    },
    "LSTM": {
        "num_layers": mai.Int(0, 4),
        "units": mai.Int(min_value=25, max_value=200),
        "activation": mai.Choice(["tanh", "sigmoid"]),
        "recurrent_activation": "sigmoid",
        "return_sequences": True,
    },
    "LSTM_output": {
        "units": mai.Int(min_value=25, max_value=200),
        "activation": mai.Choice(["tanh", "sigmoid"]),
        "recurrent_activation": "sigmoid",
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
        "input_shape": xtrain.shape[1:],
        "activation": "tanh",
        "recurrent_activation": "sigmoid",
        "return_sequences": True,
    },
    "GRU": {
        "num_layers": mai.Int(0, 4),
        "units": mai.Int(min_value=25, max_value=200),
        "activation": mai.Choice(["tanh", "sigmoid"]),
        "recurrent_activation": "sigmoid",
        "return_sequences": True,
    },
    "GRU_output": {
        "units": mai.Int(min_value=25, max_value=200),
        "activation": mai.Choice(["tanh", "sigmoid"]),
        "recurrent_activation": "sigmoid",
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
            "clipnorm": mai.Float(0.8, 1.2),
            "clipvalue": mai.Float(0.3, 0.7),
        },
        "compile_params": {
            "loss": "categorical_crossentropy",
            "metrics": ["accuracy"],
        },
        "fitting_params": {
            "batch_size": mai.Choice([8, 16, 32]),
            "epochs": 7,
            "validation_split": 0.10,
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
            "loss": "categorical_crossentropy",
            "metrics": ["accuracy"],
        },
        "fitting_params": {
            "batch_size": mai.Choice([8, 16, 32]),
            "epochs": 7,
            "validation_split": 0.10,
        },
    },
}
tuner = mai.Tuner(xtrain, ytrain, model_settings=model_settings)

# Hyperparameter tuning
configs = tuner.nn_bayesian_search(
    objective="accuracy_score",
    max_trials=50,
    cv=TimeSeriesSplit(n_splits=5),
)

# Save results to pickle
with open("configs/binary_case_1.pkl", "wb") as f:
    pickle.dump(configs, f)

# Plot convergence
plt.clf()
tuner.convergence_plot()
plt.ylim([0, 1])
plt.savefig("./figs/bc1_convergence.png", dpi=300)
