from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
import pyMAISE.settings as settings
from ._base import BaseModel


class GaussianProcess(BaseModel):
    def __init__(self, parameters: dict = None):
        default_params = {
            "kernel": (None, "BOTH", None),
            "alpha": (1e-10, "REGRESSION", None),
            "optimizer": ("fmin_l_bfgs_b", "BOTH", None),
            "n_restarts_optimizer": (0, "BOTH", None),
            "normalize_y": (False, "REGRESSION", None),
            "copy_X_train": (True, "BOTH", None),
            "n_targets": (None, "REGRESSION", None),
            "random_state": (settings.values.random_state, "BOTH", None),
            "max_iter_predict": (100, "CLASSIFICATION", lambda x: x >= 0),
            "warm_start": (False, "CLASSIFICATION", None),
            "multi_class": (
                "one_vs_rest",
                "CLASSIFICATION",
                lambda x: x in {"one_vs_rest", "one_vs_one"},
            ),
            "n_jobs": (None, "CLASSIFICATION", None),
            # Meta-Parameter: Turns
        }

        super().__init__(parameters, default_params)

    # ===========================================================
    # Methods
    def regressor(self):
        if settings.values.problem_type == settings.ProblemType.REGRESSION:
            return super()._get_regression_model(GaussianProcessRegressor)
        elif settings.values.problem_type == settings.ProblemType.CLASSIFICATION:
            return super()._get_classification_model(GaussianProcessClassifier)
