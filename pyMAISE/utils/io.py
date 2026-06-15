import os

import joblib
import pooch

_TUNING_RESULTS = pooch.create(
    path=pooch.os_cache("pyMAISE"),
    base_url="https://zenodo.org/records/20706521/files/",  # paste record ID after Zenodo upload
    registry={
        "chf_results.joblib": '1837c0ab1c3fc8083577968afc2b7b39eb541476d034e7d1d1ce14780b859c50', 
        "bwr_results.joblib": '40d17ca6d7aa684eaec99bc20196311133ace609154418e9688f4ad09a932070',
        "heat_conduction_results.joblib": '33b5b0836d30b2a67323d29c52508c3da6882788923519fb04f4124360e1fc4b',
        "fuel_performance_results.joblib": '09f34f22afdeef3d389696da6efc2037d59de8a8f9b97740b3c481db8f559905',
        "mit_reactor_results.joblib": 'af5806c5673e0369c16a728d5df30c1e4fe0c715aa3de2da62783f8f965140d0',
        "reactor_physics_results.joblib": '5895e301d9f0e9106bac859bc2f23121bdcea7bb754a837abb227dc559b6a019',
        "rod_ejection_results.joblib": 'b3f2ea770799079bc54038d67cd858c751470cc696dd3bad7d526ca18f613f1e',
        "htgr_microreactor_results_751.joblib": '9b24f8e41ad35d384c9a0ec271abb0b7859758c6a98d826f1124e50145b52049',
        "htgr_microreactor_results_3004.joblib": '76c540698904b2264a8f118240ddac7afc8c9d4fa2ae1d0204cfcc15cf65dd03',
    },
)


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
    """Load hyperparameter tuning results, fetching from Zenodo if not found locally.

    Parameters
    ----------
    filepath : str or Path
        Path to the ``.joblib`` file. If absent, the file is fetched from
        Zenodo using the basename and cached at ``pooch.os_cache("pyMAISE")``.
    tuner : pyMAISE.Tuner, optional
        If provided and a tuning state was saved, ``tuner._tuning`` is
        restored so that ``Tuner.convergence_plot`` works.

    Returns
    -------
    model_configs : list of dict
        The restored list of search-result dicts.
    """
    if not os.path.exists(filepath):
        filepath = _TUNING_RESULTS.fetch(os.path.basename(filepath))
    payload = joblib.load(filepath)
    if tuner is not None and payload.get("tuning_state") is not None:
        tuner._tuning.update(payload["tuning_state"])
    return payload["model_configs"]
