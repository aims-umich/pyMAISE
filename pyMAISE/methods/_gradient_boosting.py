from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
import pyMAISE.settings as settings
from ._base import BaseModel
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier


class GradientBoosting(BaseModel):
    def __init__(self, parameters: dict = None):
        default_params = {
            "loss": (
                "squared_error"
                if settings.values.problem_type == settings.ProblemType.REGRESSION
                else "log_loss",
                "BOTH",
                None,
            ),
            "learning_rate": (0.1, "BOTH", lambda x: x >= 0.0),
            "n_estimators": (100, "BOTH", lambda x: x >= 1),
            "subsample": (1.0, "BOTH", lambda x: 0.0 < x <= x),
            "criterion": (
                "friedman_mse",
                "BOTH",
                lambda x: x in {"friedman_mse", "squared_error"},
            ),
            "min_samples_split": (
                2,
                "BOTH",
                lambda x: (isinstance(x, float) and 0.0 < x <= 1.0)
                or (isinstance(x, int) and x >= 2),
            ),
            "min_samples_leaf": (
                1,
                "BOTH",
                lambda x: (isinstance(x, float) and 0.0 < x <= 1.0)
                or (isinstance(x, int) and x >= 1),
            ),
            "min_weight_fraction_leaf": (0.0, "BOTH", lambda x: 0.0 <= x <= 0.5),
            "max_depth": (3, "BOTH", lambda x: x >= 1),
            "min_impurity_decrease": (0.0, "BOTH", lambda x: 0.0 <= x),
            "init": (None, "BOTH", None),
            "random_state": (settings.values.random_state, "BOTH", None),
            "max_features": (None, "BOTH", None),
            "alpha": (0.9, "REGRESSION", None),
            "verbose": (0, "BOTH", lambda x: isinstance(x, int) and x >= 0),
            "max_leaf_nodes": (None, "BOTH", None),
            "warm_start": (False, "BOTH", lambda x: isinstance(x, bool)),
            "validation_fraction": (0.1, "BOTH", lambda x: 0.0 < x < 1.0),
            "n_iter_no_change": (None, "BOTH", None),
            "tol": (1e-3, "BOTH", None),
            "multi_output": (False, "NONE", lambda x: isinstance(x, bool)),
        }

        super().__init__(parameters, default_params)

    # ===========================================================
    # Methods
    def regressor(self):
        if self._multi_output:
            if settings.values.problem_type == settings.ProblemType.REGRESSION:
                return MultiOutputRegressor(
                    super()._get_regression_model(GradientBoostingRegressor)
                )
            elif settings.values.problem_type == settings.ProblemType.CLASSIFICATION:
                return MultiOutputClassifier(
                    super()._get_classification_model(GradientBoostingClassifier)
                )

        if settings.values.problem_type == settings.ProblemType.REGRESSION:
            return super()._get_regression_model(GradientBoostingRegressor)
        elif settings.values.problem_type == settings.ProblemType.CLASSIFICATION:
            return super()._get_classification_model(GradientBoostingClassifier)
