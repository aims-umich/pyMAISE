import joblib


def save_tuning_results(filepath, model_configs, tuner=None):
    """Save hyperparameter tuning results to a file.

    Parameters
    ----------
    filepath : str or Path
        Destination path (conventionally ending in ``.joblib``).
    model_configs : list of dict
        List of search-result dicts as returned by Tuner search methods.
        Each dict maps model names to ``(top_configs_DataFrame, estimator)``
        tuples.
    tuner : pyMAISE.Tuner, optional
        If provided, the convergence state used by
        ``Tuner.convergence_plot`` is also persisted.
    """
    payload = {
        "model_configs": model_configs,
        "tuning_state": dict(tuner._tuning) if tuner is not None else None,
    }
    joblib.dump(payload, filepath)


def load_tuning_results(filepath, tuner=None):
    """Load hyperparameter tuning results saved by :func:`save_tuning_results`.

    Parameters
    ----------
    filepath : str or Path
        Path to the ``.joblib`` file to read.
    tuner : pyMAISE.Tuner, optional
        If provided and a tuning state was saved, ``tuner._tuning`` is
        restored so that ``Tuner.convergence_plot`` works.

    Returns
    -------
    model_configs : list of dict
        The restored list of search-result dicts.
    """
    payload = joblib.load(filepath)
    if tuner is not None and payload.get("tuning_state") is not None:
        tuner._tuning.update(payload["tuning_state"])
    return payload["model_configs"]
