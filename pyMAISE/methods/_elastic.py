from sklearn.linear_model import ElasticNet as ElasticNetRegressor
import pyMAISE.settings as settings
from ._base import BaseModel


class ElasticNet(BaseModel):
    def __init__(self, parameters: dict = None):
        # Default parameters for Ridge regression
        default_params = {
            "alpha": (1.0, "REGRESSION", lambda x: x >= 0),
            "l1_ratio": (0.5, "REGRESSION", None),
            "fit_intercept": (True, "REGRESSION", None),
            "precompute": (False, "REGRESSION", None),
            "max_iter": (1000, "REGRESSION", lambda x: x is None or x > 0),
            "copy_X": (True, "REGRESSION", None),
            "tol": (1e-3, "REGRESSION", lambda x: x > 0),
            "warm_start": (False, "REGRESSION", None),
            "positive": (False, "REGRESSION", None),
            "random_state": (settings.values.random_state, "REGRESSION", None),
            "selection": ("cyclic", "REGRESSION", lambda x: x in {"cyclic", "random"}),
        }

        # Initialize with BaseModel
        super().__init__(parameters, default_params)

    # Implementing regressor method
    def regressor(self):
        return super()._get_regression_model(ElasticNetRegressor)
