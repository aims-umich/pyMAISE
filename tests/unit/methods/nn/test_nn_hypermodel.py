import keras_tuner
import tensorflow.keras as keras

from pyMAISE import Int
from pyMAISE.methods import nnHyperModel


def test_fnn_build():
    # Define feed forward neural network settings
    fnn_settings = {
        "structural_params": {
            "Dense_input": {
                "units": Int(min_value=25, max_value=250),
                "activation": "relu",
                "kernel_initializer": "normal",
                "sublayer": "Dropout",
                "Dropout": {
                    "rate": 0.5,
                },
            },
            "Dense_hidden": {
                "num_layers": 2,
                "units": Int(min_value=25, max_value=250),
                "activation": "relu",
                "kernel_initializer": "normal",
            },
            "Dense_output": {
                "units": 22,
                "activation": "linear",
                "kernel_initializer": "normal",
            },
        },
        "optimizer": "Adam",
        "Adam": {
            "learning_rate": 0.0001,
        },
        "compile_params": {
            "loss": "mean_absolute_error",
            "metrics": ["mean_absolute_error"],
        },
        "fitting_params": {
            "batch_size": 16,
            "epochs": 50,
            "validation_split": 0.15,
        },
    }

    # Initialize model
    pyMAISE_model = nnHyperModel(fnn_settings, input_shape=(6,), name="")

    # Build Keras model
    keras_model = pyMAISE_model.build(keras_tuner.HyperParameters())

    # Check base structure
    assert isinstance(keras_model.layers[0], keras.layers.Dense)
    assert isinstance(keras_model.layers[1], keras.layers.Dropout)
    assert isinstance(keras_model.layers[2], keras.layers.Dense)
    assert isinstance(keras_model.layers[3], keras.layers.Dense)
    assert isinstance(keras_model.layers[4], keras.layers.Dense)

    # Check hyperparameters of Dense_input
    dense_input = keras_model.layers[0].get_config()
    assert dense_input["name"] == "Dense_input_0"
    assert dense_input["units"] >= 25 and dense_input["units"] <= 250
    assert dense_input["activation"] == "relu"
    assert dense_input["kernel_initializer"]["class_name"] == "RandomNormal"

    # Check hyperparameters of Dropout sublayer
    dense_input_dropout_sublayer = keras_model.layers[1].get_config()
    assert dense_input_dropout_sublayer["name"] == "Dense_input_0_sublayer_Dropout_0"
    assert dense_input_dropout_sublayer["rate"] == 0.5

    # Check hyperparameters of Dense_hidden_0
    dense_hidden_0 = keras_model.layers[2].get_config()
    assert dense_hidden_0["name"] == "Dense_hidden_0"
    assert dense_hidden_0["units"] >= 25 and dense_hidden_0["units"] <= 250
    assert dense_hidden_0["activation"] == "relu"
    assert dense_hidden_0["kernel_initializer"]["class_name"] == "RandomNormal"

    # Check hyperparameters of Dense_hidden_1
    dense_hidden_1 = keras_model.layers[3].get_config()
    assert dense_hidden_1["name"] == "Dense_hidden_1"
    assert dense_hidden_1["units"] >= 25 and dense_hidden_1["units"] <= 250
    assert dense_hidden_1["activation"] == "relu"
    assert dense_hidden_1["kernel_initializer"]["class_name"] == "RandomNormal"

    # Check hyperparameters of Dense_output
    dense_output = keras_model.layers[4].get_config()
    assert dense_output["name"] == "Dense_output_0"
    assert dense_output["units"] == 22
    assert dense_output["activation"] == "linear"
    assert dense_output["kernel_initializer"]["class_name"] == "RandomNormal"
