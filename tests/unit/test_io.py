import os
import tempfile

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from pyMAISE.utils.io import load_tuning_results, save_tuning_results


class _MockTuner:
    def __init__(self):
        self._tuning = {}


def _make_configs():
    df = pd.DataFrame({"params": [{"alpha": 0.1}, {"alpha": 0.5}]})
    estimator = LinearRegression()
    return [{"LinearRegression": (df, estimator)}]


def test_round_trip_no_tuner():
    model_configs = _make_configs()
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
        path = f.name
    try:
        save_tuning_results(path, model_configs)
        loaded = load_tuning_results(path)
        assert len(loaded) == 1
        assert "LinearRegression" in loaded[0]
        df_loaded = loaded[0]["LinearRegression"][0]
        pd.testing.assert_frame_equal(
            df_loaded, model_configs[0]["LinearRegression"][0]
        )
    finally:
        os.unlink(path)


def test_round_trip_with_tuner_state():
    model_configs = _make_configs()
    tuner = _MockTuner()
    tuner._tuning["LR"] = np.array([[0.9, 0.85], [0.01, 0.02]])

    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
        path = f.name
    try:
        save_tuning_results(path, model_configs, tuner=tuner)

        restore_tuner = _MockTuner()
        loaded = load_tuning_results(path, tuner=restore_tuner)

        assert "LR" in restore_tuner._tuning
        np.testing.assert_array_equal(restore_tuner._tuning["LR"], tuner._tuning["LR"])
        assert len(loaded) == 1
    finally:
        os.unlink(path)


def test_load_without_tuner_ignores_state():
    model_configs = _make_configs()
    tuner = _MockTuner()
    tuner._tuning["LR"] = np.array([[0.9], [0.01]])

    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
        path = f.name
    try:
        save_tuning_results(path, model_configs, tuner=tuner)
        loaded = load_tuning_results(path)  # no tuner arg
        assert len(loaded) == 1
    finally:
        os.unlink(path)


def test_save_no_tuner_state_does_not_restore():
    model_configs = _make_configs()
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
        path = f.name
    try:
        save_tuning_results(path, model_configs, tuner=None)
        restore_tuner = _MockTuner()
        load_tuning_results(path, tuner=restore_tuner)
        assert restore_tuner._tuning == {}
    finally:
        os.unlink(path)
