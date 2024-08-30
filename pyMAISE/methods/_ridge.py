from sklearn.linear_model import Ridge as RidgeRegressor


class RidgeRegression:
    def __init__(self, parameters: dict = None):
        # Model parameters
        self._alpha = 1.0
        self._fit_intercept = True
        self._copy_X = True
        self._max_iter = None
        self._tol = 1e-3
        self._solver = "auto"
        self._positive = False
        self._random_state = None

        # Change if user provided changes in dictionary
        if parameters is not None:
            for key, value in parameters.items():
                setattr(self, key, value)

    # ===========================================================
    # Methods
    def regressor(self):
        return RidgeRegressor(
            alpha=self._alpha,
            fit_intercept=self._fit_intercept,
            copy_X=self._copy_X,
            max_iter=self._max_iter,
            tol=self._tol,
            solver=self._solver,
            positive=self._positive,
            random_state=self._random_state,
        )

    # ===========================================================
    # Getters
    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def fit_intercept(self) -> bool:
        return self._fit_intercept

    @property
    def copy_X(self) -> bool:
        return self._copy_X

    @property
    def max_iter(self) -> int:
        return self._max_iter

    @property
    def tol(self) -> float:
        return self._tol

    @property
    def solver(self) -> str:
        return self._solver

    @property
    def positive(self) -> bool:
        return self._positive

    @property
    def random_state(self):
        return self._random_state

    # ===========================================================
    # Setters
    @alpha.setter
    def alpha(self, alpha: float):
        assert alpha >= 0
        self._alpha = alpha

    @fit_intercept.setter
    def fit_intercept(self, fit_intercept: bool):
        self._fit_intercept = fit_intercept

    @copy_X.setter
    def copy_X(self, copy_X: bool):
        self._copy_X = copy_X

    @max_iter.setter
    def max_iter(self, max_iter: int):
        assert max_iter is None or max_iter > 0
        self._max_iter = max_iter

    @tol.setter
    def tol(self, tol: float):
        assert tol > 0
        self._tol = tol

    @solver.setter
    def solver(self, solver: str):
        allowed = {
            "auto",
            "svd",
            "cholesky",
            "lsqr",
            "sparse_cg",
            "sag",
            "saga",
            "lbfgs",
        }
        assert solver in allowed
        self._solver = solver

    @positive.setter
    def positive(self, positive: bool):
        self._positive = positive

    @random_state.setter
    def random_state(self, random_state):
        self._random_state = random_state
