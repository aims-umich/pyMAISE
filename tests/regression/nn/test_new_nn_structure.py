from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler

import pyMAISE as mai
from pyMAISE.datasets import load_fp, load_MITR, load_xs
from pyMAISE.preprocessing import scale_data, train_test_split


def test_new_nn_structure():
    plus_minus = 0.02

    # Loop over each base benchmark data set
    load_functions = [load_MITR, load_fp, load_xs]

    for load_function in load_functions:
        # ======================================================================
        # Old Model Structure
        global_settings = mai.init(
            problem_type=mai.ProblemType.REGRESSION,
            verbosity=1,
            random_state=42,
            num_configs_saved=1,
            cuda_visible_devices="-1",
            new_nn_architecture=False,
        )

        # Get data
        data, inputs, outputs = load_function()

        # Train test split
        xtrain, xtest, ytrain, ytest = train_test_split(
            data=[inputs, outputs], test_size=0.3
        )
        xtrain, xtest, _ = scale_data(xtrain, xtest, MinMaxScaler())
        ytrain, ytest, yscaler = scale_data(ytrain, ytest, MinMaxScaler())
        data = (xtrain, xtest, ytrain, ytest)

        # Old NN model settings
        model_settings = {
            "models": ["nn"],
            "nn": {
                # Sequential
                "num_layers": 2,
                "dropout": True,
                "rate": 0.5,
                "validation_split": 0.15,
                "loss": "mean_absolute_error",
                "metrics": ["mean_absolute_error"],
                "batch_size": 16,
                "epochs": 50,
                # Starting Layer
                "start_num_nodes": 100,
                "start_kernel_initializer": "normal",
                "start_activation": "relu",
                "input_dim": xtrain.shape[1],  # Number of inputs
                # Middle Layers
                "mid_num_node_strategy": "constant",
                "mid_kernel_initializer": "normal",
                "mid_activation": "relu",
                # Ending Layer
                "end_num_nodes": ytrain.shape[1],  # Number of outputs
                "end_activation": "linear",
                "end_kernel_initializer": "normal",
                # Optimizer
                "optimizer": "adam",
                "learning_rate": 0.0001,
            },
        }
        tuning = mai.Tuner(data[0], data[2], model_settings=model_settings)

        # Grid search space
        grid_search_space = {
            "nn": {
                # Sequantial
                "batch_size": [8, 128],
                # Starting Layer
                "start_num_nodes": [100, 400],
                # Optimizer
                "learning_rate": [0.0001, 0.001],
            }
        }

        # Grid search
        grid_search_configs = tuning.grid_search(
            param_spaces=grid_search_space,
            models=grid_search_space.keys(),
            cv=ShuffleSplit(
                n_splits=2, test_size=0.15, random_state=global_settings.random_state
            ),
        )

        new_model_settings = {"nn": {"epochs": 200}}

        # Model post-processing
        postprocessor = mai.PostProcessor(
            data=data,
            model_configs=[grid_search_configs],
            new_model_settings=new_model_settings,
            yscaler=yscaler,
        )

        old_nn_structure_results = postprocessor.metrics()

        # ======================================================================
        # New Model Structure
        global_settings.new_nn_architecture = True

        # New NN model settings
        structural = {
            "Dense_hidden": {
                "units": mai.Choice([100, 400]),
                "input_dim": xtrain.shape[1],
                "activation": "relu",
                "kernel_initializer": "normal",
                "sublayer": "Dropout",
                "Dropout": {"rate": 0.5},
            },
            "Dense_output": {
                "units": ytrain.shape[1],
                "activation": "linear",
                "kernel_initializer": "normal",
            },
        }
        model_settings = {
            "models": ["nn"],
            "nn": {
                "structural_params": structural,
                "optimizer": "Adam",
                "Adam": {
                    "learning_rate": mai.Choice([0.0001, 0.001]),
                },
                "compile_params": {
                    "loss": "mean_absolute_error",
                    "metrics": ["mean_absolute_error"],
                },
                "fitting_params": {
                    "batch_size": mai.Choice([8, 128]),
                    "epochs": 50,
                    "validation_split": 0.15,
                },
            },
        }
        tuning = mai.Tuner(data[0], data[2], model_settings=model_settings)

        # Grid search
        grid_search_configs = tuning.nn_grid_search(
            objective="r2_score",
            cv=ShuffleSplit(
                n_splits=2, test_size=0.15, random_state=global_settings.random_state
            ),
        )

        new_model_settings = {"nn": {"fitting_params": {"epochs": 200}}}

        # Model post-processing
        postprocessor = mai.PostProcessor(
            data=data,
            model_configs=[grid_search_configs],
            new_model_settings=new_model_settings,
            yscaler=yscaler,
        )

        new_nn_structure_results = postprocessor.metrics()
        postprocessor.print_model()
        print(f"Data set: {load_function}")
        print("Old Model Results\n", old_nn_structure_results.to_string())
        print("New Model Results\n", new_nn_structure_results.to_string())

        # Compare top models hyperparameters
        assert (
            old_nn_structure_results.loc[0, "Parameter Configurations"][
                "start_num_nodes"
            ]
            == new_nn_structure_results.loc[0, "Parameter Configurations"][
                "Dense_hidden_0_units"
            ]
        )
        assert (
            old_nn_structure_results.loc[0, "Parameter Configurations"]["learning_rate"]
            == new_nn_structure_results.loc[0, "Parameter Configurations"][
                "Adam_learning_rate"
            ]
        )
        assert (
            old_nn_structure_results.loc[0, "Parameter Configurations"]["batch_size"]
            == new_nn_structure_results.loc[0, "Parameter Configurations"]["batch_size"]
        )

        # Compare performance metrics
        assert (
            abs(
                old_nn_structure_results.loc[0, "Train R2"]
                - new_nn_structure_results.loc[0, "Train R2"]
            )
            / old_nn_structure_results.loc[0, "Train R2"]
            < plus_minus
        )
        assert (
            abs(
                old_nn_structure_results.loc[0, "Test R2"]
                - new_nn_structure_results.loc[0, "Test R2"]
            )
            / old_nn_structure_results.loc[0, "Test R2"]
            < plus_minus
        )


test_new_nn_structure()
