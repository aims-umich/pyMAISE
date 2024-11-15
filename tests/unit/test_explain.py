import numpy as np
import pandas as pd
from pyMAISE.explain.shap.explainers import KernelExplainer
from pyMAISE.explain.shap.explainers import GradientExplainer
from pyMAISE.explain.shap.explainers import DeepExplainer
from pyMAISE.explain.shap.explainers import ExactExplainer
from pyMAISE.explain import _explain as explain
from pyMAISE.explain.shap.plots._beeswarm import summary_legacy as summary_plot
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
import pytest

"""
Anatomy of a test:
1) arange
2) act
3) assert
4) clean up
"""

@pytest.fixture
def dummy_df():
    dummy_dict = {
        'means' :[1, 2, 3]
    }
    return pd.DataFrame(dummy_dict)

def test_plot_bar_with_labels(dummy_df):
    n_plots_before = plt.gcf().number
    explain.plot_bar_with_labels(dummy_df)
    n_plots_after = plt.gcf().number
    assert n_plots_after > n_plots_before

@pytest.fixture()
def nn_model_and_xtest():
    X, y = make_regression(n_samples=100, n_features=5, n_targets=3, noise=0.1, random_state=42)
    xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)
    # Build a dummy neural network model
    model = Sequential()
    model.add(Input(shape=(xtrain.shape[1],)))
    model.add(Dense(10, kernel_initializer='normal', activation='relu'))
    model.add(Dense(ytrain.shape[1], activation='linear', kernel_initializer='normal'))
    # Compile the model
    learning_rate = 1e-3
    model.compile(loss='mean_absolute_error', optimizer=Adam(learning_rate), metrics=['mean_absolute_error'])
    # Train the model
    model.fit(xtrain, ytrain, epochs=2, batch_size=64, validation_split=0.15, verbose=0)
    return model, xtest

@pytest.fixture()
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

def test_KernelSHAP(explain_object):
    explain_object.KernelSHAP()
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

def test_postprocess_results(explain_object):
    explain_object.KernelSHAP()
    explain_object.postprocess_results()
    assert explain_object.shap_mean["KernelSHAP"] is not None

def test_explain_plot(explain_object):
    explain_object.KernelSHAP()
    explain_object.postprocess_results()
    n_plots_before = plt.gcf().number
    explain_object.plot()
    n_plots_after = plt.gcf().number
    assert n_plots_after > n_plots_before
