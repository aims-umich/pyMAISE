from sklearn.linear_model import Ridge as RidgeRegressor
import pyMAISE.settings as settings
from ._base import BaseModel


class RidgeRegression(BaseModel):
    def __init__(self, parameters: dict = None):
        # Default parameters for Ridge regression
        default_params = {
            "alpha": (1.0, "REGRESSION", lambda x: x >= 0),
            "fit_intercept": (True, "REGRESSION", None),
            "copy_X": (True, "REGRESSION", None),
            "max_iter": (None, "REGRESSION", lambda x: x is None or x > 0),
            "tol": (1e-3, "REGRESSION", lambda x: x > 0),
            "solver": (
                "auto",
                "REGRESSION",
                lambda x: x
                in {
                    "auto",
                    "svd",
                    "cholesky",
                    "lsqr",
                    "sparse_cg",
                    "sag",
                    "saga",
                    "lbfgs",
                },
            ),
            "positive": (False, "REGRESSION", None),
            "random_state": (settings.values.random_state, "REGRESSION", None),
        }

        # Initialize with BaseModel
        super().__init__(parameters, default_params)

    # Implementing regressor method
    def regressor(self):
        return super()._get_regression_model(RidgeRegressor)
