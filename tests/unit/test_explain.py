import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from pyMAISE.explain import _explain as explain


@pytest.fixture
def dummy_df():
    dummy_dict = {"means": [1, 2, 3]}
    return pd.DataFrame(dummy_dict)


def test_plot_bar_with_labels(dummy_df):
    n_plots_before = plt.gcf().number
    explain.plot_bar_with_labels(dummy_df)
    n_plots_after = plt.gcf().number
    assert n_plots_after > n_plots_before
    plt.close("all")


@pytest.fixture(scope="module")
def nn_model_and_xtest():
    X, y = make_regression(
        n_samples=100, n_features=5, n_targets=3, noise=0.1, random_state=42
    )
    xtrain, xtest, ytrain, ytest = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = Sequential()
    model.add(Input(shape=(xtrain.shape[1],)))
    model.add(Dense(10, kernel_initializer="normal", activation="relu"))
    model.add(Dense(ytrain.shape[1], activation="linear", kernel_initializer="normal"))
    learning_rate = 1e-3
    model.compile(
        loss="mean_absolute_error",
        optimizer=Adam(learning_rate),
        metrics=["mean_absolute_error"],
    )
    model.fit(xtrain, ytrain, epochs=2, batch_size=64, validation_split=0.15, verbose=0)
    return model, xtest


@pytest.fixture(scope="module")
def explain_object(nn_model_and_xtest):
    model, xtest = nn_model_and_xtest
    return explain.ShapExplainers(model, xtest)


def test_ShapExplainers(nn_model_and_xtest, explain_object):
    model, xtest = nn_model_and_xtest
    pred_explain = explain_object.model.predict(xtest)
    pred_original = explain_object.model.predict(xtest)
    assert np.allclose(pred_explain, pred_original, 1e-3)
    assert explain_object.X.shape == xtest.shape
    assert explain_object.n_features == 5
    assert explain_object.n_outputs == 3


def test_DeepLIFT(explain_object):
    explain_object.DeepLIFT()
    assert explain_object.shap_raw["DeepLIFT"] is not None
    assert explain_object.shap_samples["DeepLIFT"] is not None


@pytest.mark.xfail(raises=AttributeError)
def test_KernelSHAP_too_many_samples(explain_object):
    explain_object.KernelSHAP()


def test_KernelSHAP(explain_object):
    explain_object.KernelSHAP(n_background_samples=3, n_test_samples=2, n_bootstrap=2)
    assert explain_object.shap_raw["KernelSHAP"] is not None
    assert explain_object.shap_samples["KernelSHAP"] is not None


def test_IntGradients(explain_object):
    explain_object.IntGradients()
    assert explain_object.shap_raw["IG"] is not None
    assert explain_object.shap_samples["IG"] is not None


def test_ExactSHAP(explain_object):
    explain_object.Exact_SHAP()
    assert explain_object.shap_raw["ExactSHAP"] is not None
    assert explain_object.shap_samples["ExactSHAP"] is not None


@pytest.fixture(scope="module")
def deeplift_explain_object(explain_object):
    explain_object.DeepLIFT()
    return explain_object


@pytest.mark.xfail(reason=AttributeError)
def test_explain_plot_without_postprocess_call(deeplift_explain_object):
    deeplift_explain_object.plot(save_figs=False)


def test_postprocess_results(deeplift_explain_object):
    deeplift_explain_object.postprocess_results()
    assert deeplift_explain_object.shap_mean["DeepLIFT"] is not None


def test_explain_plot(deeplift_explain_object):
    deeplift_explain_object.postprocess_results()
    n_plots_before = plt.gcf().number
    deeplift_explain_object.plot(save_figs=False)
    n_plots_after = plt.gcf().number
    assert n_plots_after > n_plots_before
    plt.close("all")


@pytest.mark.xfail(reason=NameError)
def test_explain_plot_output_name_error(deeplift_explain_object):
    deeplift_explain_object.plot(output_name="Fake Output Name")
