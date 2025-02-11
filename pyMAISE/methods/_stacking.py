from sklearn.ensemble import StackingRegressor, StackingClassifier
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
import pyMAISE.settings as settings
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


class Stacking:
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
        self._estimators = None
        if settings.values.problem_type == settings.ProblemType.REGRESSION:
            self._final_estimator = (
                RidgeRegression().regressor()
            )  # Default for regression
        else:
            self._final_estimator = (
                LogisticRegression().regressor()
            )  # Default for classification
        self._cv = 5
        self._n_jobs = None
        self._passthrough = False
        self._verbose = 0
        self._multi_output = False

        # Change if user provided changes in dictionary
        if parameters is not None:
            for key, value in parameters.items():
                setattr(self, key, value)

    # =======================================================
    #  Methods
    def regressor(self):
        if self._estimators is None:
            raise ValueError("You need to specify at least one estimator")
        if self._multi_output:
            if settings.values.problem_type == settings.ProblemType.REGRESSION:
                return MultiOutputRegressor(
                    StackingRegressor(
                        estimators=self._estimators,
                        final_estimator=self._final_estimator,
                        cv=self._cv,
                        n_jobs=self._n_jobs,
                        passthrough=self._passthrough,
                        verbose=self._verbose,
                    )
                )
            elif settings.values.problem_type == settings.ProblemType.CLASSIFICATION:
                return MultiOutputClassifier(
                    StackingClassifier(
                        estimators=self._estimators,
                        final_estimator=self._final_estimator,
                        cv=self._cv,
                        n_jobs=self._n_jobs,
                        passthrough=self._passthrough,
                        verbose=self._verbose,
                    )
                )

        if settings.values.problem_type == settings.ProblemType.REGRESSION:
            return StackingRegressor(
                estimators=self._estimators,
                final_estimator=self._final_estimator,
                cv=self._cv,
                n_jobs=self._n_jobs,
                passthrough=self._passthrough,
                verbose=self._verbose,
            )
        elif settings.values.problem_type == settings.ProblemType.CLASSIFICATION:
            return StackingClassifier(
                estimators=self._estimators,
                final_estimator=self._final_estimator,
                cv=self._cv,
                n_jobs=self._n_jobs,
                passthrough=self._passthrough,
                verbose=self._verbose,
            )

    def __set_estimator(self, estimator_names):
        for estimator_name in estimator_names:
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

        # If every estimator_name is well defined,
        # get the model and put it in a tuple for the estimator.
        if settings.values.problem_type == settings.ProblemType.REGRESSION:
            value = [
                (
                    estimator_name,
                    self.__valid_regression_estimators[estimator_name]().regressor(),
                )
                for estimator_name in estimator_names
            ]
        if settings.values.problem_type == settings.ProblemType.CLASSIFICATION:
            value = [
                (
                    estimator_name,
                    self.__valid_regression_estimators[estimator_name]().regressor(),
                )
                for estimator_name in estimator_names
            ]

        return value

    def __set_final_estimator(self, estimator_name):
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
    def estimators(self):
        return self._estimators

    @property
    def final_estimator(self):
        return self._final_estimator

    @property
    def cv(self):
        return self._cv

    @property
    def n_jobs(self):
        return self._n_jobs

    @property
    def passthrough(self):
        return self._passthrough

    @property
    def verbose(self):
        return self._verbose

    # ===========================================================
    # Setters
    @estimators.setter
    def estimators(self, estimators_names):
        estimators = self.__set_estimator(estimators_names)
        self._estimators = estimators

    @final_estimator.setter
    def final_estimator(self, final_estimator_name):
        final_estimator = self.__set_final_estimator(final_estimator_name)
        self._final_estimator = final_estimator

    @cv.setter
    def cv(self, cv):
        self._cv = cv

    @n_jobs.setter
    def n_jobs(self, n_jobs):
        self._n_jobs = n_jobs

    @passthrough.setter
    def passthrough(self, passthrough):
        self._passthrough = passthrough

    @verbose.setter
    def verbose(self, verbose):
        self._verbose = verbose
