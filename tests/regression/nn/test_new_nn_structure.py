import pandas as pd
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler

import pyMAISE as mai
from pyMAISE.datasets import load_MITR
from pyMAISE.methods import nnHyperModel
from pyMAISE.preprocessing import scale_data, train_test_split


def test_fnn_regression():
    """End-to-end test of the PyTorch FNN pipeline on the MITR benchmark.

    Verifies that grid search, postprocessing, and print_model all complete
    without error and produce a finite R² score.  The old architecture
    comparison (new_nn_architecture flag) is removed because NeuralNetsRegression
    has been replaced by the skorch/Optuna backend.
    """
    global_settings = mai.init(
        problem_type=mai.ProblemType.REGRESSION,
        verbosity=1,
        random_state=42,
        num_configs_saved=2,
        cuda_visible_devices="-1",
    )

    data, inputs, outputs = load_MITR()
    xtrain, xtest, ytrain, ytest = train_test_split(
        data=[inputs, outputs], test_size=0.3
    )
    xtrain, xtest, _ = scale_data(xtrain, xtest, MinMaxScaler())
    ytrain, ytest, yscaler = scale_data(ytrain, ytest, MinMaxScaler())

    structural = {
        "Dense_hidden": {
            "units": mai.Choice([64, 128]),
            "activation": "relu",
        },
        "Dense_output": {
            "units": ytrain.shape[1],
            "activation": "linear",
        },
    }
    model_settings = {
        "models": ["nn"],
        "nn": {
            "structural_params": structural,
            "optimizer": "Adam",
            "Adam": {"learning_rate": mai.Choice([0.001, 0.0001])},
            "compile_params": {"loss": "mean_absolute_error"},
            "fitting_params": {
                "batch_size": 32,
                "epochs": 10,
                "validation_split": 0.15,
            },
        },
    }
    tuner = mai.Tuner(xtrain, ytrain, model_settings=model_settings)

    grid_search_configs = tuner.nn_grid_search(
        objective="r2_score",
        cv=ShuffleSplit(
            n_splits=2, test_size=0.15, random_state=global_settings.random_state
        ),
    )

    assert isinstance(grid_search_configs["nn"][0], pd.DataFrame)
    assert isinstance(grid_search_configs["nn"][1], nnHyperModel)
    assert grid_search_configs["nn"][0].shape[0] <= global_settings.num_configs_saved

    postprocessor = mai.PostProcessor(
        data=(xtrain, xtest, ytrain, ytest),
        model_configs=[grid_search_configs],
        new_model_settings={"nn": {"fitting_params": {"epochs": 20}}},
        yscaler=yscaler,
    )

    metrics = postprocessor.metrics()
    assert metrics.shape[0] <= global_settings.num_configs_saved
    assert metrics["Train R2"].notna().all()
    assert metrics["Test R2"].notna().all()

    postprocessor.print_model()
