"""
binary_case_2.py.

Script for hyperparameter tuning LSTM and GRU using 3D data.
"""

import pickle

import matplotlib.pyplot as plt
import numpy as np
import settings
from preprocessing import load_anomaly_data, split_sequences
from sklearn.model_selection import TimeSeriesSplit
from keras.layers import TimeDistributed

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
plt.clf()
plt.bar(["Fault", "Run"], np.sum(np.concatenate([ytrain, ytest], axis=0), axis=0))
plt.title("Frequency of One-Hot Fault/Non Fault")
plt.xlabel("Categories")
plt.ylabel("Frequency")
plt.ticklabel_format(style="plain", axis="y")
plt.savefig("./figs/bc2_frequency.png", dpi=300)

# NN structure
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

cnn_lstm_structure = {
    "Reshape_input": {"target_shape": (5, 2, xtrain.shape[-1])},
    "Conv1D_input": {
        "filters": mai.Int(min_value=50, max_value=150),
        "kernel_size": mai.Int(min_value=1, max_value=5),
        "activation": "relu",
        "padding": "same",
        "wrapper": (
            TimeDistributed,
            {"input_shape": (None, 2, xtrain.shape[-1])},
        ),
    },
    "MaxPooling1D_input": {
        "pool_size": 2,
        "wrapper": TimeDistributed,
        "padding": "same",
        "sublayer": mai.Choice(["Dropout", "None"]),
        "Dropout": {
            "rate": mai.Float(min_value=0.2, max_value=0.6),
            "wrapper": TimeDistributed,
        },
    },
    "Conv1D": {
        "num_layers": mai.Int(min_value=0, max_value=3),
        "filters": mai.Int(min_value=50, max_value=150),
        "kernel_size": mai.Int(min_value=1, max_value=5),
        "activation": "relu",
        "padding": "same",
        "wrapper": TimeDistributed,
        "sublayer": "MaxPooling1D",
        "MaxPooling1D": {
            "pool_size": 2,
            "wrapper": TimeDistributed,
            "padding": "same",
            "sublayer": mai.Choice(["Dropout", "None"]),
            "Dropout": {
                "rate": mai.Float(min_value=0.2, max_value=0.6),
                "wrapper": TimeDistributed,
            },
        },
    },
    "Flatten": {
        "wrapper": TimeDistributed,
    },
    "LSTM": {
        "num_layers": mai.Int(min_value=0, max_value=4),
        "units": mai.Int(min_value=25, max_value=200),
        "activation": mai.Choice(["tanh", "sigmoid"]),
        "recurrent_activation": "sigmoid",
        "recurrent_dropout": mai.Choice([0.0, 0.2, 0.4, 0.6]),
        "return_sequences": True,
    },
    "LSTM_output": {
        "units": mai.Int(min_value=25, max_value=200),
        "activation": mai.Choice(["tanh", "sigmoid"]),
        "recurrent_activation": "sigmoid",
    },
    "Dense": {
        "num_layers": mai.Int(min_value=0, max_value=5),
        "units": mai.Int(min_value=25, max_value=300),
        "activation": "relu",
    },
    "Dense_output": {
        "units": ytrain.shape[-1],
        "activation": "sigmoid",
    },
}

model_settings = {
    "models": ["LSTM", "GRU", "CNN-LSTM"],
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
            "clipnorm": mai.Float(0.8, 1.2),
            "clipvalue": mai.Float(0.3, 0.7),
        },
        "compile_params": {
            "loss": "categorical_crossentropy",
            "metrics": ["accuracy"],
        },
        "fitting_params": {
            "batch_size": mai.Choice([32, 64, 128]),
            "epochs": 7,
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
            "loss": "categorical_crossentropy",
            "metrics": ["accuracy"],
        },
        "fitting_params": {
            "batch_size": mai.Choice([32, 64, 128]),
            "epochs": 7,
            "validation_split": 0.15,
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
with open("./configs/binary_case_2.pkl", "wb") as f:
    pickle.dump(configs, f)

# Plot convergence
plt.clf()
tuner.convergence_plot()
plt.ylim([0, 1])
plt.savefig("./figs/bc2_convergence.png", dpi=300)
