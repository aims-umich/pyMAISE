import copy
import math
import pickle

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import torch
from skorch import NeuralNetClassifier, NeuralNetRegressor
from torchview import draw_graph
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from tqdm.auto import tqdm

import pyMAISE.settings as settings
from pyMAISE.methods.nn import DeepEnsemble, DeepEnsembleHyperModel
from pyMAISE.tuner import Tuner
from pyMAISE.utils import _try_clear
from pyMAISE.utils.trial import determine_class_from_probabilities


class PostProcessor:
    """
    Assess the performance of the top-performing models.

    Parameters
    ----------
    data: tuple of xarray.DataArray
        The training and testing data given as ``(xtrain, xtest, ytrain, ytest)``.
    model_configs: single or list of dict of tuple(pandas.DataFrame, model object)
        The model configurations produced by :class:`pyMAISE.Tuner`.
    new_model_settings: dict of dict of int, float, str, or None, default=None
        Updated model settings given as a dictionary under the model's key.
    yscaler: callable or None, default=None
        An object with an ``inverse_transform`` method such as
        `min-max scaler from sklearn <https://scikit-learn.org/stable/\
        modules/generated/sklearn.preprocessing.MinMaxScaler.html>`_
        :cite:`scikit-learn`. This should have been fit using
        :meth:`pyMAISE.preprocessing.scale_data` before hyperparameter
        tuning. If ``None`` then scaling is not undone.
    """

    def __init__(
        self,
        data,
        model_configs,
        new_model_settings=None,
        yscaler=None,
    ):
        # Extract data
        self._xtrain, self._xtest, self._ytrain, self._ytest = data

        # Initialize lists
        model_types = []
        params = []
        model_wrappers = []

        # Convert to list if only one is given
        if isinstance(model_configs, dict):
            model_configs = [model_configs]

        # Extract models and the DataFrame of hyperparameter configurations
        for models in model_configs:
            for model, configs in models.items():
                # Fill model types list with string of model type
                model_types = model_types + [model] * len(configs[0]["params"])

                # Fil parameter string with hyperparameter configurations for each type
                params = params + configs[0]["params"].tolist()

                # Get all model wrappers and update parameter configurations if needed
                estimator = configs[1]
                if new_model_settings is not None and model in new_model_settings:
                    if model in Tuner.supported_classical_models:
                        estimator = estimator.set_params(**new_model_settings[model])
                    else:
                        estimator.set_params(new_model_settings[model])

                model_wrappers = model_wrappers + [estimator] * len(
                    configs[0]["params"]
                )

        # If the model is a DeepEnsemble, this sets the build method to return the entire
        # collection of trained models, opposed to just the single model required by the Tuner.
        for wrapper in model_wrappers:
            if hasattr(wrapper, "ensemble_mode"):
                wrapper.ensemble_mode = True

        # Create models DataFrame
        self._models = pd.DataFrame(
            {
                "Model Types": model_types,
                "Parameter Configurations": params,
                "Model Wrappers": model_wrappers,
            }
        )

        # If we are doing Gaussian Processing, check for standard scaling
        if "GP" in model_types:
            self.__check_if_standard_scaled(self._xtrain)
            self.__check_if_standard_scaled(self._xtest)

        # Fit each model to training data and get predicted training
        # and testing from each model
        yhat_train, yhat_test, ystd_train, ystd_test, histories = self._fit()

        # Scale predicted data if scaler is given
        self._yscaler = yscaler
        if self._yscaler is not None:
            for i in range(len(yhat_train)):
                yhat_train[i] = self._yscaler.inverse_transform(yhat_train[i])
                yhat_test[i] = self._yscaler.inverse_transform(yhat_test[i])

        # Create pandas.DataFrame
        self._models = pd.concat(
            [
                self._models,
                pd.DataFrame(
                    {
                        "Train Yhat": yhat_train,
                        "Test Yhat": yhat_test,
                        "Train Ystd": ystd_train,
                        "Test Ystd": ystd_test,
                        "History": histories,
                    }
                ),
            ],
            axis=1,
        )

        _try_clear()

    # ===========================================================
    # Methods
    def __check_if_standard_scaled(self, data, name=""):
        """Check if the data is standardized (mean ~ 0, std ~ 1)."""
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)

        mean_check = np.allclose(mean, 0, atol=1e-6)  # Check if mean is close to 0
        std_check = np.allclose(std, 1, atol=1e-6)  # Check if std is close to 1

        if not mean_check or not std_check:
            raise ValueError(
                f"{name} data is not standard scaled: " f"mean = {mean}, std = {std}"
            )

    def _fit(self):
        """Fit all models with training data and predict both training and testing
        data."""
        # Array for trainig and testing prediceted outcomes
        yhat_train = []
        yhat_test = []
        ystd_train = []
        ystd_test = []
        histories = []

        # Progress bar
        p = tqdm(
            range(self._models.shape[0]),
        )

        # Fit each model and predict outcomes
        for i in range(self._models.shape[0]):
            p.desc = self._models["Model Types"][i]
            p.n += 1
            p.refresh()

            is_classical = (
                self._models["Model Types"][i] in Tuner.supported_classical_models
            )

            if is_classical:
                # Classical sklearn models: apply params via set_params then fit.
                regressor = self._models["Model Wrappers"][i].set_params(
                    **self._models["Parameter Configurations"][i]
                )
                # Drop the last dimension when there is only one feature so sklearn
                # receives a 1-D array rather than (N, 1).
                xtrain = (
                    self._xtrain
                    if self._xtrain.shape[-1] > 1
                    else self._xtrain.isel(**{self._xtrain.dims[-1]: 0})
                )
                ytrain = (
                    self._ytrain
                    if self._ytrain.shape[-1] > 1
                    else self._ytrain.isel(**{self._ytrain.dims[-1]: 0})
                )
                regressor.fit(xtrain.values, ytrain.values)
                histories.append(None)

                # Predict (sklearn accept numpy; xarray coerces implicitly but
                # .values is explicit and safe)
                yhat_train.append(
                    regressor.predict(self._xtrain.values).reshape(
                        -1, self._ytrain.shape[-1]
                    )
                )
                yhat_test.append(
                    regressor.predict(self._xtest.values).reshape(
                        -1, self._ytest.shape[-1]
                    )
                )
                ystd_train.append(None)
                ystd_test.append(None)

            else:
                # Neural network models: reconstruct the exact trial from the
                # stored params dict using FixedTrial so build() receives the
                # same hyperparameter values that were selected during search.
                params = self._models["Parameter Configurations"][i]
                fixed_trial = optuna.trial.FixedTrial(params)
                regressor = self._models["Model Wrappers"][i].build(fixed_trial)

                # fit() now returns {"loss": [...], "val_loss": [...]} directly;
                # no .model.history.history chaining needed.
                history = self._models["Model Wrappers"][i].fit(
                    fixed_trial,
                    regressor,
                    self._xtrain.values,
                    self._ytrain.values,
                )
                histories.append(history)

                # skorch predict() accepts numpy arrays and returns numpy.
                # verbose is set at construction time (verbose=0 in build()),
                # so it is not passed here.
                if settings.values.problem_type == settings.ProblemType.REGRESSION:
                    yhat_train.append(
                        regressor.predict(self._xtrain.values).reshape(
                            -1, self._ytrain.shape[-1]
                        )
                    )
                    yhat_test.append(
                        regressor.predict(self._xtest.values).reshape(
                            -1, self._ytest.shape[-1]
                        )
                    )
                else:
                    yhat_train.append(
                        determine_class_from_probabilities(
                            regressor.predict(self._xtrain.values),
                            self._ytrain.values,
                        ).reshape(-1, self._ytrain.shape[-1])
                    )
                    yhat_test.append(
                        determine_class_from_probabilities(
                            regressor.predict(self._xtest.values),
                            self._ytest.values,
                        ).reshape(-1, self._ytest.shape[-1])
                    )

                # Store STD for uncertainty models: deep ensemble
                if hasattr(regressor, "predict_with_uncertainty"):
                    unc_train = regressor.predict_with_uncertainty(self._xtrain.values)
                    unc_test = regressor.predict_with_uncertainty(self._xtest.values)
                    ystd_train.append(np.sqrt(unc_train["epistemic_var"]))
                    ystd_test.append(np.sqrt(unc_test["epistemic_var"]))
                else:
                    ystd_train.append(None)
                    ystd_test.append(None)

        return (yhat_train, yhat_test, ystd_train, ystd_test, histories)

    def metrics(
        self, y=None, model_type=None, metrics=None, sort_by=None, direction=None
    ):
        """
        Calculate model performance of predicting output training and testing data.
        Default metrics are always evaluated depending on the
        :attr:`pyMAISE.Settings.problem_type`. For
        :attr:`pyMAISE.ProblemType.REGRESSION` problems, the default metrics are from
        :cite:`scikit-learn` and include:

        - ``R2``: `r-squared <https://scikit-learn.org/stable/modules/generated/\
          sklearn.metrics.r2_score.html#sklearn.metrics.r2_score>`_,
        - ``MAE``: `mean absolute error <https://scikit-learn.org/\
          stable/modules/generated/sklearn.metrics.mean_absolute_error.html#sklearn\
          .metrics.mean_absolute_error>`_,
        - ``MAPE``: `mean absolute percentage error <https://scikit-learn\
          .org/stable/modules/generated/sklearn.metrics.mean_absolute_per\
          centage_error.html>`_,
        - ``RMSE``: root mean squared error, the square
          root of ``MSE``,
        - ``RMSPE``: root mean squared percentage error.

        For :attr:`pyMAISE.ProblemType.CLASSIFICATION` problems, the default metrics are

        - ``Accuracy``: `accuracy <https://scikit-learn.org/stable/modules/\
          generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_\
          score>`_,
        - ``Recall``: `recall <https://scikit-learn.org/stable/modules/generated/\
          sklearn.metrics.recall_score.html#sklearn.metrics.recall_score>`_,
        - ``Precision``: `precision <https://scikit-learn.org/stable/modules/\
          generated/sklearn.metrics.precision_score.html#sklearn.metrics.precision_\
          score>`_,
        - ``F1``: `f1 <https://scikit-learn.org/stable/modules/generated/sklearn.\
          metrics.f1_score.html#sklearn.metrics.f1_score>`_.

        These metrics are evaluated for both the training and testing data sets.

        Parameters
        ----------
        y: int, str, or None, default=None
            The output to determine performance. If ``None`` then all outputs
            are used.
        model_type: str or None, default=None
            Determine the performance of this model. If ``None`` then all models are
            evaluated.
        metrics: dict of callable or None, default=None
            Dictionary of callable metrics such as `sklearn's metrics <https://scikit-\
            learn.org/stable/modules/model_evaluation.html>`_ other than those already
            default to this method. Must take two arguments: ``(y_true, y_pred)``. The
            key is used as the name in ``performance_data``.
        sort_by: str or None, default=None
            The metric to sort the return by. This should differentiate training
            and testing. For example, we can sort by ``testing mean_squared_error``.
            If ``None`` then the default is ``test r2_score`` for
            :attr:`pyMAISE.ProblemType.REGRESSION` and ``test accuracy_score``
            for :attr:`pyMAISE.ProblemType.CLASSIFICATION`.
        direction: `min`, `max`, or None
            Direction to ``sort_by``. Only required if a metric is defined in
            ``metrics`` that you want to sort the return by.

        Returns
        -------
        performance_data: pandas.DataFrame
            The performance statistics for the models for both the training and testing
            data.
        """

        # Define root mean squared error metric
        def root_mean_squared_error(y_true, y_pred):
            return math.sqrt(mean_squared_error(y_true, y_pred))

        def mai_recall_score(y_true, y_pred):
            return recall_score(y_true, y_pred, average="micro")

        def mai_precision_score(y_true, y_pred):
            return precision_score(y_true, y_pred, average="micro")

        def mai_f1_score(y_true, y_pred):
            return f1_score(y_true, y_pred, average="micro")

        def mai_mean_absolute_percentage_error(y_true, y_pred):
            return mean_absolute_percentage_error(y_true, y_pred) * 100

        def root_mean_square_percentage_error(y_true, y_pred):
            epsilon = np.finfo(np.float64).eps
            return 100 * np.mean(
                np.sqrt(
                    np.mean(
                        np.square(
                            (y_true - y_pred) / np.maximum(np.abs(y_true), epsilon)
                        ),
                        axis=0,
                    )
                )
            )

        # Get the list of y if not provided
        num_outputs = self._ytrain.shape[-1]
        if y is None:
            y = slice(0, num_outputs + 1)
        elif isinstance(y, str):
            y = np.where(self._ytrain.coords[self._ytrain.dims[-1]].to_numpy() == y)[0]

        # Scale training and testing output
        y_true = {
            "Train": (
                self._ytrain.values.reshape(-1, num_outputs)[:, y]
                if self._yscaler is None
                else self._yscaler.inverse_transform(
                    self._ytrain.values.reshape(-1, num_outputs)
                )[:, y]
            ),
            "Test": (
                self._ytest.values.reshape(-1, num_outputs)
                if self._yscaler is None
                else self._yscaler.inverse_transform(
                    self._ytest.values.reshape(-1, num_outputs)
                )[:, y]
            ),
        }

        # Get all metrics functions
        metrics = metrics if metrics is not None else {}
        if settings.values.problem_type == settings.ProblemType.REGRESSION:
            metrics = {
                "R2": r2_score,
                "MAE": mean_absolute_error,
                "MAPE": mai_mean_absolute_percentage_error,
                "RMSE": root_mean_squared_error,
                "RMSPE": root_mean_square_percentage_error,
                **metrics,
            }
        if settings.values.problem_type == settings.ProblemType.CLASSIFICATION:
            metrics = {
                "Accuracy": accuracy_score,
                "Recall": mai_recall_score,
                "Precision": mai_precision_score,
                "F1": mai_f1_score,
                **metrics,
            }

        evaluated_metrics = {
            **{f"Train {metric}": [] for metric in metrics},
            **{f"Test {metric}": [] for metric in metrics},
        }
        for i in range(self._models.shape[0]):
            for split in ["Train", "Test"]:
                # Get predicted data
                y_pred = self._models[f"{split} Yhat"][i].reshape(-1, num_outputs)[:, y]

                # Evaluate metrics
                for metric_name, func in metrics.items():
                    evaluated_metrics[f"{split} {metric_name}"].append(
                        func(y_true[split], y_pred)
                    )

        # Determine sort_by depending on problem
        sort_by = f"Test {next(iter(metrics))}" if sort_by is None else sort_by

        ascending = (
            sort_by
            not in (
                "Train R2",
                "Test R2",
                "Train Accuracy",
                "Test Accuracy",
            )
            or direction == "min"
        )

        # Place metrics into models DataFrame
        for key, value in evaluated_metrics.items():
            self._models[key] = value

        models = copy.deepcopy(
            self._models[
                ["Model Types", "Parameter Configurations"]
                + list(evaluated_metrics.keys())
            ]
        )

        # Parameter Configurations are now always plain dicts (from trial.params).
        models["Parameter Configurations"] = models["Parameter Configurations"].tolist()

        if model_type is None:
            return models.sort_values(sort_by, ascending=[ascending])
        else:
            return models[models["Model Types"] == model_type].sort_values(
                sort_by, ascending=[ascending]
            )

    def _get_idx(
        self, idx=None, model_type=None, sort_by=None, direction=None, nns_only=False
    ):
        """Get index of model in ``pandas.DataFrame`` based on model type and sort_by
        condition."""
        filter = self._models["Model Types"].unique()
        if model_type is not None:
            if not self._models["Model Types"].str.contains(model_type).any():
                raise RuntimeError(
                    f"Model {model_type} was not given to {PostProcessor.__name__}"
                )
        if nns_only:
            filter = set(filter) - set(Tuner.supported_classical_models.keys())

        # Determine sort_by depending on problem
        if sort_by is None:
            if settings.values.problem_type == settings.ProblemType.REGRESSION:
                sort_by = "Test R2"
            if settings.values.problem_type == settings.ProblemType.CLASSIFICATION:
                sort_by = "Test Accuracy"

        # Determine the index of the model in the DataFrame
        if idx is None:
            if model_type is not None:
                # If an index is not given but a model type is, get index
                # based on sort_by
                if (
                    sort_by
                    in (
                        "Train R2",
                        "Test R2",
                        "Train Accuracy",
                        "Test Accuracy",
                    )
                    or not direction == "min"
                ):
                    idx = self._models[self._models["Model Types"] == model_type][
                        sort_by
                    ].idxmax()
                else:
                    idx = self._models[self._models["Model Types"] == model_type][
                        sort_by
                    ].idxmin()
            else:
                # If an index is not given and the model type is not given,
                # return index of best in sort_by
                if (
                    sort_by
                    in (
                        "Train R2",
                        "Test R2",
                        "Train Accuracy",
                        "Test Accuracy",
                    )
                    or not direction == "min"
                ):
                    idx = self._models[self._models["Model Types"].isin(filter)][
                        sort_by
                    ].idxmax()
                else:
                    idx = self._models[self._models["Model Types"].isin(filter)][
                        sort_by
                    ].idxmin()

        return idx

    def save_models(
        self,
        num_models=10,
        idxs=None,
        model_types=None,
        sort_by=None,
        direction=None,
        directory=".",
    ):
        """
        Saves the top models. Models are names as ``<Model Type>_<Index in to metrics
        table>``.

        Parameters
        ----------
        num_models: int, default=None
            Number of models to save.
        idxs: int, list of ints, None, default=None
            The indices in the :meth:`pyMAISE.PostProcessor.metrics` pandas.DataFrame.
            If ``None``, then ``sort_by`` is used.
        model_types: str, list of str, or None, default=None
            The model name(s) to get. Will get the best model predictions based on
            ``sort_by``.
        sort_by: str or None, detault=None
            The metric to sort the pandas.DataFrame from
            :meth:`pyMAISE.PostProcessor.metrics` by. If ``None`` then
            ``test r2_score`` is used for :attr:`pyMAISE.ProblemType.REGRESSION`
            and ``test accuracy_score`` is used for
            :attr:`pyMAISE.ProblemType.CLASSIFICATION`.
        direction: 'min', 'max', or None, default=None
            The direction to ``sort_by``. It is only required if ``sort_by`` is not
            a default metric.
        directory: str, default="."
            Directory to save the models to. Classical sklearn models are saved as
            pickles (``.pkl``) and neural network models are saved as PyTorch
            checkpoints (``.pt``).
        """
        # Get indices
        if idxs:
            if isinstance(idxs, int):
                idxs = [idxs]
            elif not isinstance(idxs, list) and not isinstance(idxs, np.ndarray):
                raise TypeError(
                    "idxs must be an int, list of ints, np.ndarray of ints, or None"
                )

        # Get model_types
        if model_types:
            if isinstance(model_types, str):
                model_types = [model_types]
            elif not isinstance(model_types, list) and not isinstance(
                model_types, np.ndarray
            ):
                raise TypeError(
                    "model_types must be a string, list of strings, "
                    + "np.ndarray of strings, or None"
                )

        # Get sort if not given and ascending
        ascending = True if direction == "min" else False

        if sort_by is None:
            sort_by = (
                "Test R2"
                if settings.values.problem_type == settings.ProblemType.REGRESSION
                else "Test Accuracy"
            )

            ascending = False

        # Get indices if not given
        if idxs is None:
            # Sort models
            sorted_models = self._models[
                ["Model Types", "Parameter Configurations", "Model Wrappers", sort_by]
            ].sort_values(sort_by, ascending=ascending)

            # Filter models by model type
            if model_types:
                sorted_models = sorted_models[
                    sorted_models["Model Types"].isin(set(model_types))
                ]

            idxs = sorted_models.index.values[:num_models]

        # Iterate through idxs, train the models, and save them
        p = (
            tqdm(
                range(len(idxs)),
                desc=f"{self._models['Model Types'][idxs[0]]}_{idxs[0]}",
            )
            if settings.values.verbosity == 0
            else None
        )
        for idx in idxs:
            if p:
                p.desc = f"{self._models['Model Types'][idx]}_{idx}"
                p.refresh()

            # Train model
            model = self.get_model(idx=idx)

            # Save model: PyTorch/skorch models use torch.save (.pt);
            # classical sklearn models are pickled (.pkl).
            model_name = f"{self._models['Model Types'][idx]}_{idx}"
            if isinstance(model, (NeuralNetRegressor, NeuralNetClassifier)):
                torch.save(model, f"{directory}/{model_name}.pt")
            else:
                pickle.dump(
                    model,
                    open(f"{directory}/{model_name}.pkl", "wb"),
                )

            if p:
                p.n += 1
                p.refresh()

        if p:
            _try_clear()

    def get_predictions(self, idx=None, model_type=None, sort_by=None, direction=None):
        """
        Get a model's training and testing predictions.

        Parameters
        ----------
        idx: int or None, default=None
            The index in the :meth:`pyMAISE.PostProcessor.metrics` pandas.DataFrame.
            If ``None``, then ``sort_by`` is used.
        model_type: str or None, default=None
            The model name to get. Will get the best model predictions based on
            ``sort_by``.
        sort_by: str or None, detault=None
            The metric to sort the pandas.DataFrame from
            :meth:`pyMAISE.PostProcessor.metrics` by. If ``None`` then
            ``test r2_score`` is used for :attr:`pyMAISE.ProblemType.REGRESSION`
            and ``test accuracy_score`` is used for
            :attr:`pyMAISE.ProblemType.CLASSIFICATION`.
        direction: 'min', 'max', or None, default=None
            The direction to ``sort_by``. It is only required if ``sort_by`` is not
            a default metric.

        Returns
        -------
        yhat: tuple of numpy.array
            The predicted training and testing data given as
            ``(train_yhat, test_yhat)``.
        """
        # Determine the index of the model in the DataFrame
        idx = self._get_idx(
            idx=idx, model_type=model_type, sort_by=sort_by, direction=direction
        )

        return (self._models["Train Yhat"][idx], self._models["Test Yhat"][idx])

    def get_params(self, idx=None, model_type=None, sort_by=None, direction=None):
        """
        Returns the hyperparameters for a given model.

        Parameters
        ----------
        idx: int or None, default=None
            The index in the :meth:`pyMAISE.PostProcessor.metrics` pandas.DataFrame.
            If ``None``, then ``sort_by`` is used.
        model_type: str or None, default=None
            The model name to get. Will get the best model predictions based on
            ``sort_by``.
        sort_by: str or None, detault=None
            The metric to sort the pandas.DataFrame from
            :meth:`pyMAISE.PostProcessor.metrics` by. If ``None`` then
            ``test r2_score`` is used for :attr:`pyMAISE.ProblemType.REGRESSION`
            and ``test accuracy_score`` is used for
            :attr:`pyMAISE.ProblemType.CLASSIFICATION`.
        direction: 'min', 'max', or None, default=None
            The direction to ``sort_by``. It is only required if ``sort_by`` is not
            a default metric.

        Returns
        -------
        params: pandas.DataFrame
            The hyperparameters of the model.
        """
        # Determine the index of the model in the DataFrame
        idx = self._get_idx(
            idx=idx, model_type=model_type, sort_by=sort_by, direction=direction
        )

        # Parameter Configurations are plain dicts for both classical and NN models.
        parameters = copy.deepcopy(self._models["Parameter Configurations"][idx])
        model_type = self._models["Model Types"][idx]

        return pd.DataFrame({"Model Types": [model_type], **parameters})

    def get_model(self, idx=None, model_type=None, sort_by=None, direction=None):
        """
        Get a model. The model with the chosen hyperparameters is refit and returned.

        Parameters
        ----------
        idx: int or None, default=None
            The index in the :meth:`pyMAISE.PostProcessor.metrics` pandas.DataFrame.
            If ``None``, then ``sort_by`` is used.
        model_type: str or None, default=None
            The model name to get. Will get the best model predictions based on
            ``sort_by``.
        sort_by: str or None, detault=None
            The metric to sort the pandas.DataFrame from
            :meth:`pyMAISE.PostProcessor.metrics` by. If ``None`` then
            ``test r2_score`` is used for :attr:`pyMAISE.ProblemType.REGRESSION`
            and ``test accuracy_score`` is used for
            :attr:`pyMAISE.ProblemType.CLASSIFICATION`.
        direction: 'min', 'max', or None, default=None
            The direction to ``sort_by``. It is only required if ``sort_by`` is not
            a default metric.

        Returns
        -------
        model: sklearn estimator or skorch NeuralNet
            The model refit based on the parameters from the arguments.
        """
        # Determine the index of the model in the DataFrame
        idx = self._get_idx(
            idx=idx, model_type=model_type, sort_by=sort_by, direction=direction
        )

        # Get regressor and fit the model
        if self._models["Model Types"][idx] in Tuner.supported_classical_models:
            xtrain = (
                self._xtrain
                if self._xtrain.shape[-1] > 1
                else self._xtrain.isel(**{self._xtrain.dims[-1]: 0})
            )
            ytrain = (
                self._ytrain
                if self._ytrain.shape[-1] > 1
                else self._ytrain.isel(**{self._ytrain.dims[-1]: 0})
            )
            regressor = (
                self._models["Model Wrappers"][idx]
                .set_params(**self._models["Parameter Configurations"][idx])
                .fit(xtrain.values, ytrain.values)
            )

        else:
            # Reconstruct the trial from the stored params dict so build()
            # receives the exact hyperparameter values chosen during search.
            params = self._models["Parameter Configurations"][idx]
            fixed_trial = optuna.trial.FixedTrial(params)
            regressor = self._models["Model Wrappers"][idx].build(fixed_trial)
            self._models["Model Wrappers"][idx].fit(
                fixed_trial,
                regressor,
                self._xtrain.values,
                self._ytrain.values,
            )

        return regressor

    def diagonal_validation_plot(
        self, ax=None, y=None, idx=None, model_type=None, sort_by=None, direction=None, show_uncertainty=True,
    ):
        """
        Create a diagonal validation plot for a given model.

        Parameters
        ----------
        ax: matplotlib.pyplot.axis or None, default=None
            If not given, then an axis is created.
        y: single or list of int or str or None, default=None
            The output to plot. If ``None`` then all outputs are plotted.
        idx: int or None, default=None
            The index in the :meth:`pyMAISE.PostProcessor.metrics` pandas.DataFrame.
            If ``None``, then ``sort_by`` is used.
        model_type: str or None, default=None
            The model name to get. Will get the best model predictions based on
            ``sort_by``.
        sort_by: str or None, detault=None
            The metric to sort the pandas.DataFrame from
            :meth:`pyMAISE.PostProcessor.metrics` by. If ``None`` then
            ``test r2_score`` is used for :attr:`pyMAISE.ProblemType.REGRESSION`
            and ``test accuracy_score`` is used for
            :attr:`pyMAISE.ProblemType.CLASSIFICATION`.
        direction: 'min', 'max', or None, default=None
            The direction to ``sort_by``. It is only required if ``sort_by`` is not
            a default metric.

        Returns
        -------
        ax: matplotlib.pyplot.axis
            The plot.
        """
        # Determine the index of the model in the DataFrame
        idx = self._get_idx(
            idx=idx, model_type=model_type, sort_by=sort_by, direction=direction
        )

        # Get the list of y if not provided
        if not isinstance(y, list):
            y = [y] if y is not None else list(range(self._ytrain.shape[-1]))

        if ax is None:
            ax = plt.gca()

        ytrain = self._ytrain.values
        ytest = self._ytest.values

        if self._yscaler is not None:
            ytrain = self._yscaler.inverse_transform(
                ytrain.reshape(-1, ytrain.shape[-1])
            )
            ytest = self._yscaler.inverse_transform(ytest.reshape(-1, ytest.shape[-1]))

        for y_idx in y:
            if isinstance(y_idx, str):
                y_idx = np.where(
                    self._ytrain.coords[self._ytrain.dims[-1]].to_numpy() == y_idx
                )[0]

            ax.scatter(
                self._models["Train Yhat"][idx][..., y_idx],
                ytrain[..., y_idx],
                c="b",
                marker="o",
                s=5,
            )
            ax.scatter(
                self._models["Test Yhat"][idx][..., y_idx],
                ytest[..., y_idx],
                c="r",
                marker="o",
                s=5,
            )

            # Plotting of uncertainty error bars for supported UQ models
            if show_uncertainty and self._models["Test Ystd"][idx] is not None:
                train_ystd = self._models["Train Ystd"][idx].copy()
                test_ystd = self._models["Test Ystd"][idx].copy()
                if self._yscaler is not None:
                    train_ystd = train_ystd / self._yscaler.scale_
                    test_ystd = test_ystd / self._yscaler.scale_

                ax.errorbar(
                    self._models["Train Yhat"][idx][..., y_idx],
                    ytrain[..., y_idx],
                    yerr=train_ystd[..., y_idx],
                    fmt="none",
                    ecolor="b",
                    alpha=0.5,
                    capsize=0,
                    elinewidth=1,
                )
                ax.errorbar(
                    self._models["Test Yhat"][idx][..., y_idx],
                    ytest[..., y_idx],
                    yerr=test_ystd[..., y_idx],
                    fmt="none",
                    ecolor="r",
                    alpha=0.5,
                    capsize=0,
                    elinewidth=1,
                )

        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()]),
        ]

        ax.plot(lims, lims, "k--")
        ax.set_aspect("equal")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.legend(["Training Data", "Testing Data"])
        ax.set_xlabel("Predicted Outcome")
        ax.set_ylabel("Actual Outcome")

        return ax

    def validation_plot(
        self,
        ax=None,
        y=None,
        idx=None,
        model_type=None,
        sort_by=None,
        direction=None,
    ):
        """
        Create a validation plot for a given model.

        Parameters
        ----------
        ax: matplotlib.pyplot.axis or None, default=None
            If not given, then an axis is created.
        y: single or list of int or str or None, default=None
            The output to plot. If ``None`` then all outputs are plotted.
        idx: int or None, default=None
            The index in the :meth:`pyMAISE.PostProcessor.metrics` pandas.DataFrame.
            If ``None``, then ``sort_by`` is used.
        model_type: str or None, default=None
            The model name to get. Will get the best model predictions based on
            ``sort_by``.
        sort_by: str or None, detault=None
            The metric to sort the pandas.DataFrame from
            :meth:`pyMAISE.PostProcessor.metrics` by. If ``None`` then
            ``test r2_score`` is used for :attr:`pyMAISE.ProblemType.REGRESSION`
            and ``test accuracy_score`` is used for
            :attr:`pyMAISE.ProblemType.CLASSIFICATION`.
        direction: 'min', 'max', or None, default=None
            The direction to ``sort_by``. It is only required if ``sort_by`` is not
            a default metric.

        Returns
        -------
        ax: matplotlib.pyplot.axis
            The plot.
        """
        # Determine the index of the model in the DataFrame
        idx = self._get_idx(
            idx=idx, model_type=model_type, sort_by=sort_by, direction=direction
        )

        # Get the list of y if not provided
        if not isinstance(y, list):
            y = [y] if y is not None else list(range(self._ytrain.shape[-1]))

        # Get prediected and actual outputs
        ytest = self._ytest.values
        yhat_test = self._models["Test Yhat"][idx]

        if self._yscaler is not None:
            ytest = self._yscaler.inverse_transform(ytest.reshape(-1, ytest.shape[-1]))

        if ax is None:
            ax = plt.gca()

        for y_idx in y:
            # If the column name is given as opposed to the position,
            # find the position
            if isinstance(y_idx, str):
                y_idx = np.where(
                    self._ytest.coords[self._ytest.dims[-1]].values == y_idx
                )[0]

            ax.scatter(
                np.linspace(1, ytest.shape[0], ytest.shape[0]),
                np.abs((ytest[:, y_idx] - yhat_test[:, y_idx]) / ytest[:, y_idx]) * 100,
                label=self._ytest.coords[self._ytest.dims[-1]].values[y_idx],
            )

        if len(y) > 1:
            ax.legend()

        ax.set_xlabel("Testing Data Index")
        ax.set_ylabel("Absolute Relative Error (%)")

        return ax

    def nn_learning_plot(
        self, ax=None, idx=None, model_type=None, sort_by=None, direction=None
    ):
        """
        Create a learning plot for a given neural network.

        Parameters
        ----------
        ax: matplotlib.pyplot.axis or None, default=None
            If not given then an axis is created.
        idx: int or None, default=None
            The index in the :meth:`pyMAISE.PostProcessor.metrics` pandas.DataFrame.
            If ``None``, then ``sort_by`` is used.
        model_type: str or None, default=None
            The model name to get. Will get the best model predictions based on
            ``sort_by``.
        sort_by: str or None, detault=None
            The metric to sort the pandas.DataFrame from
            :meth:`pyMAISE.PostProcessor.metrics` by. If ``None`` then
            ``test r2_score`` is used for :attr:`pyMAISE.ProblemType.REGRESSION`
            and ``test accuracy_score`` is used for
            :attr:`pyMAISE.ProblemType.CLASSIFICATION`.
        direction: 'min', 'max', or None, default=None
            The direction to ``sort_by``. It is only required if ``sort_by`` is not
            a default metric.

        Returns
        -------
        ax: matplotlib.pyplot.axis
            The plot.
        """
        # Determine the index of the model in the DataFrame
        idx = self._get_idx(
            idx=idx,
            model_type=model_type,
            sort_by=sort_by,
            direction=direction,
            nns_only=True,
        )

        ax = ax or plt.gca()

        history = self._models["History"][idx]

        ax.plot(history["loss"], label="Training")
        ax.plot(history["val_loss"], label="Validation")
        ax.legend()
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")

        return ax

    def print_model(self, idx=None, model_type=None, sort_by=None, direction=None):
        """
        Print a models tuned hyperparameters.

        Parameters
        ----------
        idx: int or None, default=None
            The index in the :meth:`pyMAISE.PostProcessor.metrics` pandas.DataFrame.
            If ``None``, then ``sort_by`` is used.
        model_type: str or None, default=None
            The model name to get. Will get the best model predictions based on
            ``sort_by``.
        sort_by: str or None, default=None
            The metric to sort the pandas.DataFrame from
            :meth:`pyMAISE.PostProcessor.metrics` by. If ``None`` then
            ``test r2_score`` is used for :attr:`pyMAISE.ProblemType.REGRESSION`
            and ``test accuracy_score`` is used for
            :attr:`pyMAISE.ProblemType.CLASSIFICATION`.
        direction: 'min', 'max', or None, default=None
            The direction to ``sort_by``. It is only required if ``sort_by`` is not
            a default metric.
        """
        # Determine the index of the model in the DataFrame
        idx = self._get_idx(
            idx=idx,
            model_type=model_type,
            sort_by=sort_by,
            direction=direction,
            nns_only=False,
        )

        # Get model parameters
        params = self.get_params(idx=idx).to_dict()

        # Print parameters if not NN else ensure only pertinent information is printed
        print(f"Model Type: {params.pop('Model Types')[0]}")
        if model_type in Tuner.supported_classical_models:
            for key, value in params.items():
                print(f"  {key}: {value[0]}")
        else:
            # Rebuild the model with FixedTrial to get the PyTorch module structure.
            stored_params = self._models["Parameter Configurations"][idx]
            fixed_trial = optuna.trial.FixedTrial(stored_params)
            model = self._models["Model Wrappers"][idx].build(fixed_trial)

            # Print tuned hyperparameters grouped by whether they belong to a
            # structural layer or to compile/fitting configuration.
            print("  Structural Hyperparameters")
            for key, value in params.items():
                is_layer_param = any(
                    layer_name in key
                    for layer_name in self._models["Model Wrappers"][idx].layer_dict
                )
                if is_layer_param:
                    print(f"    {key}: {value[0]}")

            print("  Compile/Fitting Hyperparameters")
            for key, value in params.items():
                is_layer_param = any(
                    layer_name in key
                    for layer_name in self._models["Model Wrappers"][idx].layer_dict
                )
                if not is_layer_param:
                    print(f"    {key}: {value[0]}")

            # Print the PyTorch module structure (equivalent of Keras .summary()).
            # initialize() materializes module_ without requiring a full fit.
            model.initialize()
            print("\n  Module Structure")
            print(model.module_)

    def nn_network_plot(
        self, idx=None, model_type=None, sort_by=None, direction=None, **kwargs
    ):
        """
        Plot NN network architecture using torchview.

        .. note::

           Graphviz must be installed on your system (e.g. ``brew install graphviz``
           or ``apt install graphviz``). The ``torchview`` Python package is installed
           automatically with pyMAISE.

        Parameters
        ----------
        idx: int or None, default=None
            The index in the :meth:`pyMAISE.PostProcessor.metrics` pandas.DataFrame.
            If ``None``, then ``sort_by`` is used.
        model_type: str or None, default=None
            The model name to get. Will get the best model predictions based on
            ``sort_by``.
        sort_by: str or None, default=None
            The metric to sort the pandas.DataFrame from
            :meth:`pyMAISE.PostProcessor.metrics` by. If ``None`` then
            ``test r2_score`` is used for :attr:`pyMAISE.ProblemType.REGRESSION`
            and ``test accuracy_score`` is used for
            :attr:`pyMAISE.ProblemType.CLASSIFICATION`.
        direction: 'min', 'max', or None, default=None
            The direction to ``sort_by``. It is only required if ``sort_by`` is not
            a default metric.
        kwargs:
            Any keyword arguments forwarded to
            `torchview.draw_graph \
            <https://github.com/mert-kurttutan/torchview>`_
            except ``model`` and ``input_size``.

        Returns
        -------
        graph: graphviz.Digraph
            The rendered graph. In a Jupyter notebook this renders inline;
            call ``graph.render(filename)`` to save to a file.
        """
        idx = self._get_idx(
            idx=idx,
            model_type=model_type,
            sort_by=sort_by,
            direction=direction,
            nns_only=True,
        )

        # Rebuild the skorch model from stored hyperparameters and initialize it
        # (initialize() materializes module_ without requiring a full fit).
        params = self._models["Parameter Configurations"][idx]
        fixed_trial = optuna.trial.FixedTrial(params)
        model = self._models["Model Wrappers"][idx].build(fixed_trial)
        model.initialize()

        # Derive input shape from training data (drop the sample/batch dimension).
        input_size = tuple(self._xtrain.shape[1:])

        try:
            return draw_graph(
                model.module_, input_size=input_size, **kwargs
            ).visual_graph
        except Exception as e:
            if "dot" in str(e).lower() or "executable" in str(e).lower():
                raise RuntimeError(
                    "Graphviz system package is required for nn_network_plot(). "
                    "Install it with:\n"
                    "  conda install -c conda-forge graphviz\n"
                    "or\n"
                    "  brew install graphviz   # macOS\n"
                    "  apt install graphviz    # Ubuntu/Debian"
                ) from e
            raise

    def confusion_matrix(
        self,
        axs=None,
        idx=None,
        model_type=None,
        sort_by=None,
        direction=None,
        colorbar=False,
        annotate=True,
        round=2,
    ):
        """
        Create training and testing confusion matrix.

        Parameters
        ----------
        axs: list of 2 matplotlib.pyplot.axis or None, default=None
            If not given then an axes are created.
        idx: int or None, default=None
            The index in the :meth:`pyMAISE.PostProcessor.metrics` pandas.DataFrame.
            If ``None``, then ``sort_by`` is used.
        model_type: str or None, default=None
            The model name to get. Will get the best model predictions based on
            ``sort_by``.
        sort_by: str or None, detault=None
            The metric to sort the pandas.DataFrame from
            :meth:`pyMAISE.PostProcessor.metrics` by. If ``None`` then
            ``test r2_score`` is used for :attr:`pyMAISE.ProblemType.REGRESSION`
            and ``test accuracy_score`` is used for
            :attr:`pyMAISE.ProblemType.CLASSIFICATION`.
        direction: 'min', 'max', or None, default=None
            The direction to ``sort_by``. It is only required if ``sort_by`` is not
            a default metric.
        colorbar: Boolean, default=False
            Whether to include a colorbar.
        annotate: Boolean, default=True
            Whether to include annotations (number and percentage).
        round: int, default=2
            Number of digits to round percentage in annotation.

        Returns
        -------
        axs: tuple of matplotlib.pyplot.axis
            The two confusion matrix axes: ``(cm_train, cm_test)``
        """
        # Determine the index of the model in the DataFrame
        idx = self._get_idx(
            idx=idx, model_type=model_type, sort_by=sort_by, direction=direction
        )

        # Labels
        labels = []
        for label in self._ytrain.coords[self._ytrain.dims[-1]].values:
            labels.append(label.split("_", 1)[1])

        # Get predicted and actual outputs
        yhat_train = self._models["Train Yhat"][idx]
        yhat_test = self._models["Test Yhat"][idx]
        ytrain = self._ytrain.values
        ytest = self._ytest.values

        # Convert one-hot encoding to multilabel
        if ytrain.shape[-1] > 1:
            yhat_train = np.argmax(yhat_train, axis=-1)
            yhat_test = np.argmax(yhat_test, axis=-1)
            ytrain = np.argmax(ytrain, axis=-1)
            ytest = np.argmax(ytest, axis=-1)

        # Confusion matrix
        train_cm = confusion_matrix(y_true=ytrain, y_pred=yhat_train)
        test_cm = confusion_matrix(y_true=ytest, y_pred=yhat_test)

        # Axes
        if axs is None:
            _, axs = plt.subplots(1, 2)

        # Confusion matrix display
        ConfusionMatrixDisplay(confusion_matrix=train_cm, display_labels=labels).plot(
            include_values=False, ax=axs[0], colorbar=colorbar
        )
        ConfusionMatrixDisplay(confusion_matrix=test_cm, display_labels=labels).plot(
            include_values=False, ax=axs[1], colorbar=colorbar
        )

        # Add values and percentages
        if annotate:
            for (i, j), value in np.ndenumerate(train_cm):
                axs[0].text(
                    i,
                    j,
                    f"{value}\n{np.round(value / np.sum(train_cm) * 100, round)}%",
                    ha="center",
                    va="center",
                )

                value = test_cm[i, j]
                axs[1].text(
                    i,
                    j,
                    f"{value}\n{np.round(value / np.sum(test_cm) * 100, round)}%",
                    ha="center",
                    va="center",
                )

        axs[0].set_title("Training Set")
        axs[1].set_title("Testing Set")

        return axs
