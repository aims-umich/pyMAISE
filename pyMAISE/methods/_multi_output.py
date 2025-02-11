import pyMAISE.settings as settings
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
from pyMAISE.methods import (
    SVM,
    DecisionTree,
    KNeighbors,
    LassoRegression,
    LinearRegression,
    LogisticRegression,
    RandomForest,
    GaussianProcess,
    RidgeRegression,
    GradientBoosting,
    ElasticNet,
    AdaBoost,
    ExtraTrees,
)


class MultiOutput:
    __valid_classification_estimators = {
        "Logistic": LogisticRegression,
        "SVM": SVM,
        "DT": DecisionTree,
        "RF": RandomForest,
        "KN": KNeighbors,
        "GP": GaussianProcess,
        "GB": GradientBoosting,
        "ET": ExtraTrees,
        "AB": AdaBoost,
    }

    __valid_regression_estimators = {
        "Linear": LinearRegression,
        "Lasso": LassoRegression,
        "SVM": SVM,
        "DT": DecisionTree,
        "RF": RandomForest,
        "KN": KNeighbors,
        "GP": GaussianProcess,
        "RD": RidgeRegression,
        "GB": GradientBoosting,
        "EN": ElasticNet,
        "ET": ExtraTrees,
        "AB": AdaBoost,
    }

    def __init__(self, parameters: dict = None):
        # Model Parameters
        self._estimator = None
        self._n_jobs = None

        # Change if user provided changes in dictionary
        if parameters is not None:
            for key, value in parameters.items():
                setattr(self, key, value)

    # ===========================================================
    # Methods
    def regressor(self):
        if self._estimator is None:
            raise ValueError("Please set an estimator to use Multi Output")
        if settings.values.problem_type == settings.ProblemType.REGRESSION:
            return MultiOutputRegressor(self._estimator)
        elif settings.values.problem_type == settings.ProblemType.CLASSIFICATION:
            return MultiOutputClassifier(self._estimator)

    def __set__estimator(self, estimator_name):
        if (
            settings.values.problem_type == settings.ProblemType.REGRESSION
            and estimator_name not in self.__valid_regression_estimators
        ):
            raise ValueError(
                f"{estimator_name} is an invalid key. "
                + "Please give a valid key for this problem type"
            )
        if (
            settings.values.problem_type == settings.ProblemType.CLASSIFICATION
            and estimator_name not in self.__valid_classification_estimators
        ):
            raise ValueError(
                f"{estimator_name} is an invalid key. "
                + "Please give a valid key for this problem type"
            )

        if settings.values.problem_type == settings.ProblemType.REGRESSION:
            value = self.__valid_regression_estimators[estimator_name]().regressor()
        if settings.values.problem_type == settings.ProblemType.CLASSIFICATION:
            value = self.__valid_regression_estimators[estimator_name]().regressor()

        return value

    # ========================================================
    # Getters
    @property
    def estimator(self):
        return self._estimator

    @property
    def n_jobs(self):
        return self._n_jobs

    # ===========================================================
    # Setters
    @estimator.setter
    def estimator(self, estimator_name):
        estimator = self.__set__estimator(estimator_name)
        self._estimator = estimator

    @n_jobs.setter
    def n_jobs(self, n_jobs):
        self._n_jobs = n_jobs
