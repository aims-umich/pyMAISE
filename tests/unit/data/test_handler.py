import pytest
import xarray as xr

import pyMAISE as mai
from pyMAISE.datasets import load_chf, load_HTGR, load_loca, load_xs


@pytest.mark.datasets
def test_load_xs():
    mai.init(problem_type=mai.ProblemType.REGRESSION)

    data, inputs, outputs = load_xs()

    assert isinstance(data, xr.DataArray)
    assert isinstance(inputs, xr.DataArray)
    assert isinstance(outputs, xr.DataArray)

    assert data.shape == (1000, 9)
    assert inputs.shape == (1000, 8)
    assert outputs.shape == (1000, 1)

    input_features = [
        "FissionFast",
        "CaptureFast",
        "FissionThermal",
        "CaptureThermal",
        "Scatter12",
        "Scatter11",
        "Scatter21",
        "Scatter22",
    ]
    output_features = ["k"]
    assert list(data.coords["variable"].to_numpy()) == input_features + output_features
    assert list(inputs.coords["variable"].to_numpy()) == input_features
    assert list(outputs.coords["variable"].to_numpy()) == output_features

    assert data[0, 0] == 0.00644620
    assert data[0, -1] == 1.256376
    assert data[-1, 0] == 0.00627230
    assert data[-1, -1] == 1.240064


@pytest.mark.datasets
def test_load_loca():
    mai.init(problem_type=mai.ProblemType.REGRESSION)

    nominal_data, perturbed_data = load_loca(stack_series=False)

    assert nominal_data.shape == (1, 400, 44)
    assert perturbed_data.shape == (2000, 400, 44)


@pytest.mark.datasets
def test_load_chf():
    mai.init(problem_type=mai.ProblemType.REGRESSION)

    train_data, xtrain, ytrain, test_data, xtest, ytest = load_chf()

    for obj in (train_data, xtrain, ytrain, test_data, xtest, ytest):
        assert isinstance(obj, xr.DataArray)

    assert train_data.shape == (2000, 7)
    assert xtrain.shape == (2000, 6)
    assert ytrain.shape == (2000, 1)

    assert test_data.shape == (500, 7)
    assert xtest.shape == (500, 6)
    assert ytest.shape == (500, 1)

    input_features = ["D (m)", "L (m)", "P (kPa)", "G (kg m-2s-1)", "Tin (C)", "Xe (-)"]
    output_features = ["CHF (kW m-2)"]
    assert list(xtrain.coords["variable"].to_numpy()) == input_features
    assert list(ytrain.coords["variable"].to_numpy()) == output_features
    assert list(xtest.coords["variable"].to_numpy()) == input_features
    assert list(ytest.coords["variable"].to_numpy()) == output_features


@pytest.mark.datasets
def test_load_HTGR():
    mai.init(problem_type=mai.ProblemType.REGRESSION)

    data, inputs, outputs = load_HTGR()

    assert isinstance(data, xr.DataArray)
    assert isinstance(inputs, xr.DataArray)
    assert isinstance(outputs, xr.DataArray)

    assert inputs.shape == (751, 8)
    assert outputs.shape == (751, 4)
