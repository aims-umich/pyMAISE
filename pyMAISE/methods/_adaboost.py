from sklearn.ensemble import AdaBoostRegressor, AdaBoostClassifier
import pyMAISE.settings as settings
from ._base import BaseModel
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier


class AdaBoost(BaseModel):
    def __init__(self, parameters: dict = None):
        # Define default parameters
        default_params = {
            "estimator": (None, "BOTH", None),
            "n_estimators": (50, "BOTH", lambda x: x >= 1),
            "learning_rate": (1.0, "BOTH", lambda x: x > 0.0),
            "loss": ("linear", "REGRESSION", None),
            "random_state": (settings.values.random_state, "BOTH", None),
            "algorithm": ("SAMME.R", "CLASSIFICATION", None),
            "multi_output": (False, "NONE", lambda x: isinstance(x, bool)),
        }

        super().__init__(parameters=parameters, default_params=default_params)

    def regressor(self):
        if self._multi_output:
            if settings.values.problem_type == settings.ProblemType.REGRESSION:
                return MultiOutputRegressor(
                    super()._get_regression_model(AdaBoostRegressor)
                )
            elif settings.values.problem_type == settings.ProblemType.CLASSIFICATION:
                return MultiOutputClassifier(
                    super()._get_classification_model(AdaBoostClassifier)
                )

        if settings.values.problem_type == settings.ProblemType.REGRESSION:
            return super()._get_regression_model(AdaBoostRegressor)
        elif settings.values.problem_type == settings.ProblemType.CLASSIFICATION:
            return super()._get_classification_model(AdaBoostClassifier)
