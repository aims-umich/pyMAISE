
"""
binary_case_1.py.

inputs  (samples, time steps, features)
outputs (samples, labels)

TODO:
- Save convergence plot to ./figs/
- Add further hyperparameter tuning to LSTM
- Add GRU

See how non_faulty_frac and timestep_step change the results.
Increase number of trials and cross-validation splits once
you're feeling good about everything.
"""

import pickle
from sklearn.model_selection import TimeSeriesSplit

import pyMAISE as mai
import settings
from preprocessing import load_anomaly_data

print("\nBinary case 1")

# Initialize pyMAISE
mai.init(
    problem_type=settings.problem_type,
    verbosity=settings.verbosity,
    random_state=settings.random_state,
    cuda_visible_devices="1",  # Use GPU 1
)

# Load training/testing data
xtrain, xtest, ytrain, ytest, _ = load_anomaly_data(
    stack_series=False,
    multiclass=False,
    test_size=settings.test_size,
    non_faulty_frac=settings.non_faulty_frac,
    timestep_step=1,
)





import numpy as np
import matplotlib.pyplot as plt
frequency = np.sum(ytrain, axis=0)

# Categories
categories = ['Fault', 'NonFault']

# Plot
plt.bar(categories, frequency)
plt.title('Frequency of One-Hot Fault/Non Fault')
plt.xlabel('Categories')
plt.ylabel('Frequency')
plt.ticklabel_format(style='plain', axis='y')
plt.show()
ytrain






# Model initialization
lstm_structure = {
    "LSTM_input": {
        "units": mai.Int(min_value = 25,max_value = 75),
        "input_shape": xtrain.shape[1:],
        "activation": "tanh",
        "recurrent_activation": "sigmoid",
        "return_sequences": True,
    },
    "LSTM": {
        "num_layers": mai.Int(0, 3),
        "units": mai.Int(min_value = 25,max_value = 75),
        "activation": mai.Choice(["tanh", "sigmoid"]),
        "recurrent_activation": "sigmoid",
        "return_sequences": True,
    },
    "LSTM_output": {
        "units": mai.Int(min_value = 25,max_value = 75),
        "activation": mai.Choice(["tanh", "sigmoid"]),
        "recurrent_activation": "sigmoid",
    },
    "Dense": {
        "num_layers": mai.Int(0, 3),
        "units": mai.Int(min_value = 25,max_value = 75),
        "activation": "relu",
    },
    "Dense_output": {
        "units": ytrain.shape[-1],
        "activation": "sigmoid",
    },
}


gru_structure = {
    "GRU_input": {
        "units":  mai.Int(min_value = 25,max_value = 75),
        "input_shape": xtrain.shape[1:],
        "activation": "tanh",
        "recurrent_activation": "sigmoid",
        "return_sequences": True,
    },
    "GRU": {
        "num_layers": mai.Int(0, 3),
        "units": mai.Int(min_value = 25,max_value = 75),
        "activation": mai.Choice(["tanh", "sigmoid"]),
        "recurrent_activation": "sigmoid",
        "return_sequences": True,
    },
    "GRU_output": {
        "units": mai.Int(min_value = 25,max_value = 75),
        "activation": mai.Choice(["tanh", "sigmoid"]),
        "recurrent_activation": "sigmoid",
    },
    "Dense": {
        "num_layers": mai.Int(0, 3),
        "units": mai.Int(min_value = 25,max_value = 75),
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
        "Adam": {"learning_rate": mai.Float(1e-5, 0.001)},
        "compile_params": {
            "loss": "binary_crossentropy",
            "metrics": ["accuracy"],
        },
        "fitting_params": {
            "batch_size": mai.Choice([32, 64, 128]),
            "epochs": 2,
            "validation_split": 0.15,
        },
    },
    "GRU": {
        "structural_params": lstm_structure,
        "optimizer": "Adam",
        "Adam": {"learning_rate": mai.Float(1e-5, 0.001)},
        "compile_params": {
            "loss": "binary_crossentropy",
            "metrics": ["accuracy"],
        },
        "fitting_params": {
            "batch_size": mai.Choice([32, 64, 128]),
            "epochs": 2,
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




data_to_save = {
    'xtrain': xtrain,
    'ytrain': ytrain,
    'xtest': xtest,
    'ytest': ytest,
}

with open("/home/jacc/pyMAISE/docs/source/benchmarks/supporting/anomaly/configs/data_file.pkl", "wb") as f:
    pickle.dump(data_to_save, f)











# Save results to pickle
with open("/home/jacc/pyMAISE/docs/source/benchmarks/supporting/anomaly/configs/binary_case_1.pkl", "wb") as f:
    pickle.dump(configs, f)





ax = tuner.convergence_plot(model_types="LSTM")
ax.set_ylim([0, 1])



file_path = "/home/jacc/pyMAISE/docs/source/benchmarks/supporting/anomaly/configs/convergence_plot_lstm.png"


ax.figure.savefig(file_path)


plt.close(ax.figure)


ax = tuner.convergence_plot(model_types="GRU")
ax.set_ylim([0, 1])



file_path = "/home/jacc/pyMAISE/docs/source/benchmarks/supporting/anomaly/configs/convergence_plot_gru.png"

ax.figure.savefig(file_path)

plt.close(ax.figure)