"""
preprocessing.py.

Script for processing anomaly data.
"""

import numpy as np
import xarray as xr
from sklearn.preprocessing import MinMaxScaler

from pyMAISE.datasets import load_anomaly
from pyMAISE.preprocessing import (
    train_test_split,
    SplitSequence,
    scale_data,
    one_hot_encode,
)
import settings


def load_anomaly_data(
    global_settings,
    stack_series,
    multiclass,
    test_size,
    non_faulty_frac=1.0,
    timestep_step=1,
):
    # Set random state temporarily to get the same split data between runs
    global_settings.random_state = settings.data_random_state

    # Load data
    inputs, outputs = load_anomaly(
        input_path=settings.input_path,
        output_path=settings.output_path,
        stack_series=False,
        multiclass=multiclass,
        propagate_output=stack_series,
        non_faulty_frac=non_faulty_frac,
        timestep_step=timestep_step,
    )

    # One-hot-encode outputs
    outputs = one_hot_encode(outputs)

    # Train-test split (70% training, 30% testing)
    xtrain, xtest, ytrain, ytest = train_test_split(
        [inputs, outputs], test_size=test_size
    )

    # Reset random state
    global_settings.random_state = settings.random_state

    # Scale inputs
    xtrain, xtest, xscaler = scale_data(xtrain, xtest, MinMaxScaler())

    print(f"xtrain shape: {xtrain.shape}")
    print(f"xtest shape: {xtest.shape}")
    print(f"ytrain shape: {ytrain.shape}")
    print(f"ytest shape: {ytest.shape}")

    return xtrain, xtest, ytrain, ytest, xscaler


def split_sequences(data, input_steps, output_steps, output_position):
    xtrain, xtest, ytrain, ytest = data

    # Reshape training and testing data
    xtrain = xr.DataArray(
        xtrain.values.reshape((-1, xtrain.shape[-1])),
        coords={
            "time steps": np.arange(xtrain.shape[0] * xtrain.shape[1]),
            "features": xtrain.coords["features"].values,
        },
    )
    xtest = xr.DataArray(
        xtest.values.reshape((-1, xtest.shape[-1])),
        coords={
            "time steps": np.arange(xtest.shape[0] * xtest.shape[1]),
            "features": xtest.coords["features"].values,
        },
    )

    ytrain = xr.DataArray(
        ytrain.values.reshape((-1, ytrain.shape[-1])),
        coords={
            "time steps": np.arange(xtrain.shape[0]),
            "features": ytrain.coords["features"].values,
        },
    )
    ytest = xr.DataArray(
        ytest.values.reshape((-1, ytest.shape[-1])),
        coords={
            "time steps": np.arange(xtest.shape[0]),
            "features": ytest.coords["features"].values,
        },
    )

    # Combine data and create rolling windows using SplitSequence
    split_sequence = SplitSequence(
        input_steps=input_steps,
        output_steps=output_steps,
        output_position=output_position,
        sequence_inputs=xtrain.coords["features"].values,
        sequence_outputs=ytrain.coords["features"].values,
    )

    xtrain, ytrain = split_sequence.split(xr.concat([xtrain, ytrain], dim="features"))
    xtest, ytest = split_sequence.split(xr.concat([xtest, ytest], dim="features"))

    return xtrain, xtest, ytrain, ytest
