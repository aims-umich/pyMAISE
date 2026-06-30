import numpy as np
import xarray as xr
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import ShuffleSplit

import pyMAISE as mai
from pyMAISE import PostProcessor
from pyMAISE.preprocessing import train_test_split, one_hot_encode
from pyMAISE.methods.nn import DeepEnsemble


def simulate_regression_data():
    """Simulate small random dataset for regression testing."""
    X, y = make_regression(n_samples=50, n_features=2, noise=0.1, random_state=42)
    x_raw = xr.DataArray(X)
    y_raw = xr.DataArray(y.reshape(-1, 1))
    return train_test_split([x_raw, y_raw], test_size=0.3)


def simulate_classification_data():
    """Simulate small random dataset for classification testing."""
    X, y = make_classification(n_samples=50, n_features=4, n_classes=2, random_state=42)
    x_raw = xr.DataArray(X, dims=["samples", "features"])
    # y needs to be an object for one_hot_encode
    y_raw = xr.DataArray(y.reshape(-1, 1), dims=["samples", "variables"]).astype("object")
    y_raw.coords["variables"] = ["class"]
    y_enc = one_hot_encode(y_raw)
    return train_test_split([x_raw, y_enc], test_size=0.3)


def test_deep_ensemble_regression():
    """Test DeepEnsemble end-to-end on a regression problem."""
    global_settings = mai.init(
        problem_type=mai.ProblemType.REGRESSION, 
        random_state=42, 
        num_configs_saved=1, 
        verbosity=0
    )
    
    data = simulate_regression_data()
    xtrain, xtest, ytrain, ytest = data

    parameters = {
        "models": ["DeepEnsemble"],
        "DeepEnsemble": {
            "num_models": 3,
            "structural_params": {
                "Dense_1": {
                    "units": mai.Choice([16, 32]),
                    "activation": "relu",
                },
                "Dropout_1": {
                    "rate": 0.2,
                },
                "Dense_2": {
                    "units": ytrain.shape[-1],
                    "activation": "linear",
                }
            },
            "optimizer": "Adam",
            "Adam": {
                "learning_rate": mai.Choice([1e-3, 1e-2]),
            },
            "compile_params": {
                "loss": "mean_absolute_error",
            },
            "fitting_params": {
                "epochs": 2,
                "batch_size": 16,
            }
        }
    }

    tuner = mai.Tuner(xtrain, ytrain, model_settings=parameters)
    results = tuner.nn_grid_search(
        objective="r2_score", 
        cv=ShuffleSplit(n_splits=2, test_size=0.2, random_state=global_settings.random_state)
    )
    
    post_processor = PostProcessor(data=data, model_configs=[results])
    metrics = post_processor.metrics()
    model = post_processor.get_model(model_type="DeepEnsemble")

    assert isinstance(model, DeepEnsemble)
    assert len(model.ensemble_models) == 3
    assert metrics["Test R2"].notna().all()

    # Test predicting with uncertainty
    uncertainty_results = model.predict_with_uncertainty(xtest.values)
    assert "predictions" in uncertainty_results
    assert "mean" in uncertainty_results
    assert "epistemic_var" in uncertainty_results
    assert "aleatoric_var" in uncertainty_results
    
    assert uncertainty_results["predictions"].shape == (3, xtest.shape[0], ytrain.shape[-1])
    assert uncertainty_results["mean"].shape == (xtest.shape[0], ytrain.shape[-1])
    assert uncertainty_results["epistemic_var"].shape == (xtest.shape[0], ytrain.shape[-1])
    assert uncertainty_results["aleatoric_var"] is None

    # NLL Test
    parameters["DeepEnsemble"]["compile_params"]["loss"] = "nll"
    # # Need double the outputs for heteroscedastic NLL
    parameters["DeepEnsemble"]["structural_params"]["Dense_2"]["units"] = ytrain.shape[-1] * 2
    tuner_nll = mai.Tuner(xtrain, ytrain, model_settings=parameters)
    results_nll = tuner_nll.nn_grid_search(objective="r2_score", cv=2)
    pp_nll = PostProcessor(data=data, model_configs=[results_nll])
    pp_nll.metrics()
    model_nll = pp_nll.get_model(model_type="DeepEnsemble")
    uncert_nll = model_nll.predict_with_uncertainty(xtest.values)
    assert uncert_nll["aleatoric_var"] is not None
    assert uncert_nll["aleatoric_var"].shape == (xtest.shape[0], ytrain.shape[-1])


def test_deep_ensemble_classification():
    """Test DeepEnsemble end-to-end on a classification problem."""
    global_settings = mai.init(
        problem_type=mai.ProblemType.CLASSIFICATION, 
        random_state=42, 
        num_configs_saved=1, 
        verbosity=0
    )
    
    data = simulate_classification_data()
    xtrain, xtest, ytrain, ytest = data

    parameters = {
        "models": ["DeepEnsemble"],
        "DeepEnsemble": {
            "num_models": 2, # Test a different number of models
            "structural_params": {
                "Dense_1": {
                    "units": mai.Choice([16, 32]),
                    "activation": "relu",
                },
                "Dropout_1": {
                    "rate": mai.Choice([0.1, 0.2]),
                },
                "Dense_2": {
                    "units": ytrain.shape[-1], # 2 classes from one-hot encoding
                    "activation": "softmax",
                }
            },
            "optimizer": "Adam",
            "Adam": {
                "learning_rate": mai.Choice([1e-3, 1e-2]),
            },
            "compile_params": {
                "loss": "categorical_crossentropy",
            },
            "fitting_params": {
                "epochs": 2,
                "batch_size": 16,
            }
        }
    }

    tuner = mai.Tuner(xtrain, ytrain, model_settings=parameters)
    results = tuner.nn_grid_search(
        objective="accuracy_score", 
        cv=ShuffleSplit(n_splits=2, test_size=0.2, random_state=global_settings.random_state)
    )
    
    post_processor = PostProcessor(data=data, model_configs=[results])
    metrics = post_processor.metrics()
    model = post_processor.get_model(model_type="DeepEnsemble")

    assert isinstance(model, DeepEnsemble)
    assert len(model.ensemble_models) == 2
    assert metrics["Test Accuracy"].notna().all()

    # Test predicting with uncertainty
    uncertainty_results = model.predict_with_uncertainty(xtest.values)
    assert uncertainty_results["predictions"].shape == (2, xtest.shape[0], ytrain.shape[-1])
    assert uncertainty_results["mean"].shape == (xtest.shape[0], ytrain.shape[-1])
    # Epistemic var for classification is 1D (entropy over classes)
    assert uncertainty_results["epistemic_var"].shape == (xtest.shape[0],)
    assert uncertainty_results["aleatoric_var"] is None


def test_deep_ensemble_hyperparameter_propagation():
    """Verify that the ensemble members are built using the best hyperparameters found by the tuner."""
    global_settings = mai.init(
        problem_type=mai.ProblemType.REGRESSION, 
        random_state=42, 
        num_configs_saved=1, 
        verbosity=0
    )
    
    data = simulate_regression_data()
    xtrain, xtest, ytrain, ytest = data

    parameters = {
        "models": ["DeepEnsemble"],
        "DeepEnsemble": {
            "num_models": 2,
            "structural_params": {
                "Dense_1": {
                    "units": mai.Choice([16, 64]),
                    "activation": "relu",
                },
                "Dense_2": {
                    "units": ytrain.shape[-1],
                    "activation": "linear",
                }
            },
            "optimizer": "Adam",
            "Adam": {
                "learning_rate": 0.001,
            },
            "compile_params": {
                "loss": "mean_absolute_error",
            },
            "fitting_params": {
                "epochs": 1,
                "batch_size": 16,
            }
        }
    }

    tuner = mai.Tuner(xtrain, ytrain, model_settings=parameters)
    results = tuner.nn_grid_search(
        objective="r2_score", 
        cv=ShuffleSplit(n_splits=2, test_size=0.2, random_state=global_settings.random_state)
    )
    
    post_processor = PostProcessor(data=data, model_configs=[results])
    _ = post_processor.metrics()
    model = post_processor.get_model(model_type="DeepEnsemble")
    
    # Retrieve the chosen parameter dictionary from post_processor.get_params()
    best_params_df = post_processor.get_params(model_type="DeepEnsemble")
    
    # Identify the key corresponding to Dense_1 units.
    units_col = [c for c in best_params_df.columns if "Dense_1" in c and "units" in c][0]
    chosen_units = int(best_params_df[units_col].values[0])
    
    # Verify that the chosen units are either 16 or 64
    assert chosen_units in [16, 64]
    
    # Verify that each model in the ensemble has indeed been built with this value
    assert len(model.ensemble_models) == 2
    for member in model.ensemble_models:
        # member.module_ is a _SequentialNet, and member.module_.net is nn.Sequential
        # The first layer is _DenseBlock. We can access its 'linear' attribute
        first_layer = member.module_.net[0]
        assert hasattr(first_layer, "linear")
        
        # Check output features of the linear layer
        assert first_layer.linear.out_features == chosen_units


def test_deep_ensemble_postprocessor():
    """Test all PostProcessor methods on DeepEnsemble."""
    global_settings = mai.init(
        problem_type=mai.ProblemType.REGRESSION, 
        random_state=42, 
        num_configs_saved=1, 
        verbosity=0
    )
    
    data = simulate_regression_data()
    xtrain, xtest, ytrain, ytest = data

    parameters = {
        "models": ["DeepEnsemble"],
        "DeepEnsemble": {
            "num_models": 2,
            "structural_params": {
                "Dense_1": {
                    "units": 8,
                    "activation": "relu",
                },
                "Dense_2": {
                    "units": ytrain.shape[-1],
                    "activation": "linear",
                }
            },
            "optimizer": "Adam",
            "Adam": {
                "learning_rate": 1e-3,
            },
            "compile_params": {
                "loss": "mean_absolute_error",
            },
            "fitting_params": {
                "epochs": 1,
                "batch_size": 16,
            }
        }
    }

    tuner = mai.Tuner(xtrain, ytrain, model_settings=parameters)
    results = tuner.nn_grid_search(
        objective="r2_score", 
        cv=ShuffleSplit(n_splits=2, test_size=0.2, random_state=global_settings.random_state)
    )
    
    post_processor = PostProcessor(data=data, model_configs=[results])

    # Test metrics()
    metrics = post_processor.metrics()
    assert metrics is not None
    assert "Test MAE" in metrics.columns

    # Test get_predictions()
    train_preds, test_preds = post_processor.get_predictions(model_type="DeepEnsemble")
    assert train_preds.shape == (xtrain.shape[0], ytrain.shape[-1])
    assert test_preds.shape == (xtest.shape[0], ytest.shape[-1])

    # Test get_params()
    params = post_processor.get_params(model_type="DeepEnsemble")
    assert params is not None

    # Test get_model()
    model = post_processor.get_model(model_type="DeepEnsemble")
    assert isinstance(model, DeepEnsemble)

    # Test print_model()
    post_processor.print_model(model_type="DeepEnsemble")

    # Test diagonal_validation_plot()
    ax_diag = post_processor.diagonal_validation_plot(model_type="DeepEnsemble")
    assert ax_diag is not None

    # Test validation_plot()
    ax_val = post_processor.validation_plot(model_type="DeepEnsemble")
    assert ax_val is not None

    # Test nn_learning_plot()
    ax_learn = post_processor.nn_learning_plot(model_type="DeepEnsemble")
    assert ax_learn is not None


def test_deep_ensemble_pickle():
    """ Verify DeepEnsemble can be saved and reloaded via pickle with no custom save/load methods."""
    import pickle

    mai.init(
        problem_type=mai.ProblemType.REGRESSION,
        random_state=42,
        num_configs_saved=1,
        verbosity=0,
    )

    data = simulate_regression_data()
    xtrain, xtest, ytrain, ytest = data

    parameters = {
        "models": ["DeepEnsemble"],
        "DeepEnsemble": {
            "num_models": 2,
            "structural_params": {
                "Dense_1": {
                    "units": 8,
                    "activation": "relu",
                },
                "Dense_2": {
                    "units": ytrain.shape[-1],
                    "activation": "linear",
                },
            },
            "optimizer": "Adam",
            "Adam": {
                "learning_rate": 1e-3,
            },
            "compile_params": {
                "loss": "mean_absolute_error",
            },
            "fitting_params": {
                "epochs": 1,
                "batch_size": 16,
            },
        },
    }

    tuner = mai.Tuner(xtrain, ytrain, model_settings=parameters)
    results = tuner.nn_grid_search(objective="r2_score", cv=2)

    post_processor = PostProcessor(data=data, model_configs=[results])
    post_processor.metrics()
    model = post_processor.get_model(model_type="DeepEnsemble")

    # Predictions before picking
    original_preds = model.predict(xtest.values)
    original_uncertainty = model.predict_with_uncertainty(xtest.values)

    # Round-trip through pickle
    pickled = pickle.dumps(model)
    reloaded_model = pickle.loads(pickled)

    assert isinstance(reloaded_model, DeepEnsemble)
    assert len(reloaded_model.ensemble_models) == len(model.ensemble_models)

    # Reloaded predictions must match the original exactly
    reloaded_preds = reloaded_model.predict(xtest.values)
    np.testing.assert_allclose(reloaded_preds, original_preds)

    reloaded_uncertainty = reloaded_model.predict_with_uncertainty(xtest.values)
    np.testing.assert_allclose(
        reloaded_uncertainty["mean"], original_uncertainty["mean"]
    )
    np.testing.assert_allclose(
        reloaded_uncertainty["epistemic_var"], original_uncertainty["epistemic_var"]
    )

    # module_ property (added for skorch-style compatibility) should also
    # survive the round-trip, since downstream PostProcessor code relies on it.
    assert reloaded_model.module_ is not None


if __name__ == "__main__":
    test_deep_ensemble_regression()
    test_deep_ensemble_classification()
    test_deep_ensemble_hyperparameter_propagation()
    test_deep_ensemble_postprocessor()
    test_deep_ensemble_pickle()
