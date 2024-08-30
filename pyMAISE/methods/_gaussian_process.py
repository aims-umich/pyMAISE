from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
import pyMAISE.settings as settings


class GaussianProcess:
    def __init__(self, parameters: dict = None):
        # Model parameters
        self._kernel = None
        self._alpha = 1e-10
        self._optimizer = "fmin_l_bfgs_b"
        self._n_restarts_optimizer = 0
        self._normalize_y = False
        self._copy_X_train = True
        self._n_targets = None
        self._random_state = settings.values.random_state
        self._max_iter_predict = 100
        self._multi_class = "one_vs_rest"
        self._num_jobs = None

        # Change if user provided changes in dictionary
        if parameters is not None:
            for key, value in parameters.items():
                setattr(self, key, value)

    # ===========================================================
    # Methods
    def regressor(self):
        if settings.values.problem_type == settings.ProblemType.REGRESSION:
            return GaussianProcessRegressor(
                kernel=self._kernel,
                alpha=self._alpha,
                optimizer=self._optimizer,
                n_restarts_optimizer=self._n_restarts_optimizer,
                normalize_y=self._normalize_y,
                copy_X_train=self._copy_X_train,
                n_targets=self._n_targets,
                random_state=self._random_state,
            )
        elif settings.values.problem_type == settings.ProblemType.CLASSIFICATION:
            return GaussianProcessClassifier(
                kernel=self._kernel,
                optimizer=self._optimizer,
                n_restarts_optimizer=self._n_restarts_optimizer,
                normalize_y=self._normalize_y,
                copy_X_train=self._copy_X_train,
                n_targets=self._n_targets,
                random_state=self._random_state,
            )

    # ===========================================================
    # Getters
    @property
    def kernel(self):
        return self._kernel

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def n_restarts_optimizer(self) -> int:
        return self._n_restarts_optimizer

    @property
    def normalize_y(self) -> bool:
        return self._normalize_y

    @property
    def copy_X_train(self) -> bool:
        return self._copy_X_train

    @property
    def n_targets(self) -> int:
        return self._n_targets

    @property
    def random_state(self):
        return self._random_state

    # ===========================================================
    # Setters
    @kernel.setter
    def kernel(self, kernel):
        self._kernel = kernel

    @alpha.setter
    def alpha(self, alpha: float):
        assert alpha >= 0
        self._alpha = alpha

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    @n_restarts_optimizer.setter
    def n_restarts_optimizer(self, n_restarts_optimizer: int):
        assert n_restarts_optimizer >= 0
        self._n_restarts_optimizer = n_restarts_optimizer

    @normalize_y.setter
    def normalize_y(self, normalize_y: bool):
        self._normalize_y = normalize_y

    @copy_X_train.setter
    def copy_X_train(self, copy_X_train: bool):
        self._copy_X_train = copy_X_train

    @n_targets.setter
    def n_targets(self, n_targets: int):
        assert n_targets >= 0
        self.n_targets = n_targets

    @random_state.setter
    def random_state(self, random_state):
        self._random_state = random_state
