from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
import pyMAISE.settings as settings
from ._base import BaseModel


class ExtraTrees(BaseModel):
    def __init__(self, parameters: dict = None):
        # Define default parameters
        default_params = {
            # Common parameters for both ExtraTreesRegressor and ExtraTreesClassifier
            "n_estimators": (100, "BOTH", lambda x: x >= 1),
            "criterion": (
                "squared_error"
                if settings.values.problem_type == settings.ProblemType.REGRESSION
                else "gini",
                "BOTH",
                lambda x: x
                in ["squared_error", "absolute_error", "friedman_mse", "poisson"]
                if settings.values.problem_type == settings.ProblemType.REGRESSION
                else x in ["gini", "entropy", "log_loss"],
            ),
            "max_depth": (None, "BOTH", lambda x: x is None or x >= 1),
            "min_samples_split": (2, "BOTH", lambda x: x >= 2),
            "min_samples_leaf": (1, "BOTH", lambda x: x >= 1),
            "min_weight_fraction_leaf": (0.0, "BOTH", lambda x: 0.0 <= x <= 0.5),
            "max_features": (1.0, "BOTH", lambda x: x == "auto" or 0 < x <= 1.0),
            "max_leaf_nodes": (None, "BOTH", lambda x: x is None or x >= 1),
            "min_impurity_decrease": (0.0, "BOTH", lambda x: x >= 0),
            "bootstrap": (False, "BOTH", lambda x: isinstance(x, bool)),
            "oob_score": (False, "BOTH", lambda x: isinstance(x, bool)),
            "n_jobs": (None, "BOTH", lambda x: x is None or isinstance(x, int)),
            "random_state": (settings.values.random_state, "BOTH", None),
            "verbose": (0, "BOTH", lambda x: isinstance(x, int) and x >= 0),
            "warm_start": (False, "BOTH", lambda x: isinstance(x, bool)),
            "ccp_alpha": (0.0, "BOTH", lambda x: x >= 0),
            "max_samples": (None, "BOTH", lambda x: x is None or 0 < x <= 1.0),
            # Specific parameters for ExtraTreesClassifier
            "class_weight": (None, "CLASSIFICATION", None),
            # Specific parameters for ExtraTreesRegressor
        }

        super().__init__(parameters=parameters, default_params=default_params)

    def regressor(self):
        if settings.values.problem_type == settings.ProblemType.REGRESSION:
            return super()._get_regression_model(ExtraTreesRegressor)
        elif settings.values.problem_type == settings.ProblemType.CLASSIFICATION:
            return super()._get_classification_model(ExtraTreesClassifier)
