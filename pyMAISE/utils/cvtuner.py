import keras_tuner as kt
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.utils.multiclass import type_of_target
from tensorflow.keras.backend import clear_session
from tqdm.auto import tqdm

import pyMAISE.settings as settings


class CVTuner(kt.Tuner):
    def __init__(
        self,
        oracle,
        objective=None,
        cv=5,
        shuffle=True,
        hypermodel=None,
        max_model_size=None,
        optimizer=None,
        loss=None,
        metrics=None,
        distribution_strategy=None,
        directory=None,
        project_name=None,
        logger=None,
        tuner_id=None,
        overwrite=False,
        executions_per_trial=1,
        verbose=0,
        **kwargs,
    ):
        self._objective = objective
        self._cv = cv
        self._metrics = metrics
        self._shuffle = shuffle
        self._mean_test_score = []
        self._std_test_score = []
        self._verbose = verbose

        # Build base keras tuner
        kt.Tuner.__init__(
            self,
            oracle=oracle,
            hypermodel=hypermodel,
            max_model_size=max_model_size,
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            distribution_strategy=distribution_strategy,
            directory=directory,
            project_name=project_name,
            logger=logger,
            tuner_id=tuner_id,
            overwrite=overwrite,
            executions_per_trial=executions_per_trial,
            **kwargs,
        )

        # Progress bar
        if self._verbose == 0:
            n_fits = cv if isinstance(cv, int) else cv.n_splits
            if self.oracle.max_trials:
                n_fits *= oracle.max_trials
            else:
                for hp in self.hypermodel.get_hyperparameters():
                    n_fits *= len(hp.values)

            self._p = tqdm(
                range(int(n_fits)),
                desc=self.hypermodel._name,
            )
        else:
            print(f"Tuning {self.hypermodel._name}")

    def run_trial(self, trial, x, y, *fit_args, **fit_kwargs):
        # Reassign CV depending on what's given
        if isinstance(self._cv, int):
            if (
                settings.values.problem_type == settings.ProblemType.CLASSIFICATION
                and (type_of_target(y) in ("binary", "multiclass"))
            ):
                self._cv = StratifiedKFold(
                    n_splits=self._cv,
                    shuffle=self._shuffle,
                    random_state=(
                        settings.values.random_state if self._shuffle is True else None
                    ),
                )
            else:
                self._cv = KFold(
                    n_splits=self._cv,
                    shuffle=self._shuffle,
                    random_state=(
                        settings.values.random_state if self._shuffle is True else None
                    ),
                )

        # Run
        test_scores = []
        model = None
        for train_indices, val_indices in self._cv.split(x, y):
            # Create training and validation split based on samples dimension
            # (assumed to be the first dimension)
            x_train, x_val = x[train_indices,], x[val_indices,]
            y_train, y_val = y[train_indices,], y[val_indices,]

            # Build and fit model to training data
            model = self.hypermodel.build(trial.hyperparameters)
            self.hypermodel.fit(
                trial.hyperparameters,
                model,
                x_train,
                y_train,
            )

            # Evaluate model performance
            if self._metrics is not None:
                y_val_pred = model.predict(x_val, verbose=settings.values.verbosity)

                # Round probabilities to correct class based on data format
                if settings.values.problem_type == settings.ProblemType.CLASSIFICATION:
                    y_val_pred = determine_class_from_probabilities(y_val_pred, y)

                test_scores.append(
                    self._metrics(
                        y_val_pred.reshape(-1, y_val.shape[-1]),
                        y_val.reshape(-1, y_val.shape[-1]),
                    )
                )

            else:
                test_score_idx = model.metrics_names.index(self._objective)
                test_scores.append(model.evaluate(x_val, y_val)[test_score_idx])

            if self._verbose == 0:
                self._p.n += 1
                self._p.refresh()

        # Reset tensorflow session to reduce RAM usage
        clear_session()

        # Append performance data for CV results
        self._mean_test_score.append(np.average(test_scores))
        self._std_test_score.append(np.std(test_scores))

        # Update oracle on objective outcome
        return {self._objective: np.average(test_scores)}

    # Getters
    @property
    def mean_test_score(self):
        return self._mean_test_score

    @property
    def std_test_score(self):
        return self._std_test_score


def determine_class_from_probabilities(y_pred, y):
    if type_of_target(y) == "binary" or type_of_target(y) == "multiclass":
        # Round to nearest number
        return np.round(y_pred)

    elif type_of_target(y) == "multilabel-indicator":
        assert np.all((y_pred <= 1) & (y_pred >= 0))
        # Assert 0 or 1 for one hot encoding
        return np.where(y_pred == y_pred.max(axis=-1, keepdims=True), 1, 0)
