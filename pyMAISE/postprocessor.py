import copy
import math

import keras_tuner as kt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

import pyMAISE.settings as settings
from pyMAISE.tuner import Tuner
from pyMAISE.utils.cvtuner import determine_class_from_probabilities


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
                    if (
                        model in Tuner.supported_classical_models
                        or not settings.values.new_nn_architecture
                    ):
                        estimator = estimator.set_params(**new_model_settings[model])

                    else:
                        estimator.set_params(new_model_settings[model])

                model_wrappers = model_wrappers + [estimator] * len(
                    configs[0]["params"]
                )

        # Create models DataFrame
        self._models = pd.DataFrame(
            {
                "Model Types": model_types,
                "Parameter Configurations": params,
                "Model Wrappers": model_wrappers,
            }
        )

        # Fit each model to training data and get predicted training
        # and testing from each model
        yhat_train, yhat_test, histories = self._fit()

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
                        "History": histories,
                    }
                ),
            ],
            axis=1,
        )

    # ===========================================================
    # Methods
    def _fit(self):
        """Fit all models with training data and predict both training and testing
        data."""
        # Array for trainig and testing prediceted outcomes
        yhat_train = []
        yhat_test = []
        histories = []

        # Fit each model and predict outcomes
        for i in range(self._models.shape[0]):
            # Extract regressor for the configuration
            regressor = None
            if (
                self._models["Model Types"][i] in Tuner.supported_classical_models
                or not settings.values.new_nn_architecture
            ):
                regressor = self._models["Model Wrappers"][i].set_params(
                    **self._models["Parameter Configurations"][i]
                )
            else:
                regressor = self._models["Model Wrappers"][i].build(
                    self._models["Parameter Configurations"][i]
                )

            # Append learning curve history of neural networks and run fit for all
            if self._models["Model Types"][i] in Tuner.supported_classical_models:
                # Change final dimension if there is only one feature
                # in any of these arrays
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
            else:
                if not settings.values.new_nn_architecture:
                    histories.append(
                        regressor.fit(
                            self._xtrain.values,
                            self._ytrain.values,
                        ).model.history.history
                    )
                else:
                    histories.append(
                        self._models["Model Wrappers"][i]
                        .fit(
                            self._models["Parameter Configurations"][i],
                            regressor,
                            self._xtrain.values,
                            self._ytrain.values,
                        )
                        .model.history.history
                    )
                    if (
                        settings.values.problem_type
                        == settings.ProblemType.CLASSIFICATION
                    ):
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
                        continue

            # Append training and testing predictions
            yhat_train.append(
                regressor.predict(self._xtrain).reshape(-1, self._ytrain.shape[-1])
            )
            yhat_test.append(
                regressor.predict(self._xtest).reshape(-1, self._ytest.shape[-1])
            )

        return (yhat_train, yhat_test, histories)

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
        - ``MSE``: `mean squared error <https://scikit-learn.org/stable/\
          modules/generated/sklearn.metrics.mean_squared_error.html#sklearn.metrics.\
          mean_squared_error>`_,
        - ``RMSE``: root mean squared error, the square
          root of ``mean_squared_error``.

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
            return recall_score(y_true, y_pred, average="macro")

        def mai_precision_score(y_true, y_pred):
            return precision_score(y_true, y_pred, average="macro")

        def mai_f1_score(y_true, y_pred):
            return f1_score(y_true, y_pred, average="macro")

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
                "MSE": mean_squared_error,
                "RMSE": root_mean_squared_error,
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

        hyperparams = []
        for i in range(models.shape[0]):
            if isinstance(models["Parameter Configurations"][i], kt.HyperParameters):
                hyperparams.append(models["Parameter Configurations"][i].values)
            else:
                hyperparams.append(models["Parameter Configurations"][i])

        models["Parameter Configurations"] = hyperparams

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

        # Get values from pyMAISE.HyperParameters
        parameters = copy.deepcopy(self._models["Parameter Configurations"][idx])
        if (
            self._models["Model Types"][idx] not in Tuner.supported_classical_models
            and settings.values.new_nn_architecture
        ):
            parameters = parameters.values
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
        model: sklearn or keras model
            The model refit based on the parameters from the arguments.
        """
        # Determine the index of the model in the DataFrame
        idx = self._get_idx(
            idx=idx, model_type=model_type, sort_by=sort_by, direction=direction
        )

        # Get regressor and fit the model
        regressor = None
        if (
            self._models["Model Types"][idx] in Tuner.supported_classical_models
            or not settings.values.new_nn_architecture
        ):
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
                .fit(xtrain, ytrain)
            )

        else:
            regressor = self._models["Model Wrappers"][idx].build(
                self._models["Parameter Configurations"][idx]
            )
            regressor._name = self._models["Model Types"][idx]

            self._models["Model Wrappers"][idx].fit(
                self._models["Parameter Configurations"][idx],
                regressor,
                self._xtrain.values,
                self._ytrain.values,
            )

        return regressor

    def diagonal_validation_plot(
        self, ax=None, y=None, idx=None, model_type=None, sort_by=None, direction=None
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
            )
            ax.scatter(
                self._models["Test Yhat"][idx][..., y_idx],
                ytest[..., y_idx],
                c="r",
                marker="o",
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

            ax.plot(
                np.linspace(1, ytest.shape[0], ytest.shape[0]),
                np.abs((ytest[:, y_idx] - yhat_test[:, y_idx]) / ytest[:, y_idx]) * 100,
                "-o",
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

    def confusion_matrix(
        self,
        axs=None,
        fig=None,
        figsize=None,
        idx=None,
        model_type=None,
        sort_by=None,
        direction=None,
    ):
        """
        Create training and testing confusion matrix.

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
        axs: tuple of matplotlib.pyplot.axis
            The two confusion matrix axes: ``(cm_train, cm_test)``
        """
        # Determine the index of the model in the DataFrame
        idx = self._get_idx(
            idx=idx, model_type=model_type, sort_by=sort_by, direction=direction
        )

        # Get predicted and actual outputs
        yhat_train = self._models["Train Yhat"][idx]
        yhat_test = self._models["Test Yhat"][idx]

        ytrain = self._ytrain.values
        ytest = self._ytest.values

        if self._yscaler is not None:
            ytrain = self._yscaler.inverse_transform(
                ytrain.reshape(-1, ytrain.shape[-1])
            )
            ytest = self._yscaler.inverse_transform(ytest.reshape(-1, ytest.shape[-1]))

        if axs is None or fig is None:
            fig, axs = plt.subplots(
                ytrain.shape[-1] if ytrain.shape[-1] > 2 else 1, 2, figsize=figsize
            )

        if ytrain.shape[-1] > 2:
            for i in range(axs.shape[0]):
                # Training plot
                disp = ConfusionMatrixDisplay(
                    confusion_matrix(ytrain[..., i], yhat_train[..., i]),
                )
                disp.plot(ax=axs[i, 0], values_format=".4g")
                class_label = (
                    self._ytrain.coords[self._ytrain.dims[-1]]
                    .values[i]
                    .replace("_", " ")
                )
                disp.ax_.set_title(f"Train {class_label}")
                disp.im_.colorbar.remove()

                if i != axs.shape[0] - 1:
                    disp.ax_.set_xlabel("")

                # Testing plot
                disp = ConfusionMatrixDisplay(
                    confusion_matrix(ytest[..., i], yhat_test[..., i]),
                )
                disp.plot(ax=axs[i, 1], values_format=".4g")
                class_label = (
                    self._ytest.coords[self._ytest.dims[-1]].values[i].replace("_", " ")
                )
                disp.ax_.set_ylabel("")
                disp.ax_.set_title(f"Test {class_label}")
                disp.im_.colorbar.remove()

                if i != axs.shape[0] - 1:
                    disp.ax_.set_xlabel("")
                else:
                    plt.subplots_adjust(wspace=0.10, hspace=0.1)
                    fig.colorbar(disp.im_, ax=axs)

        else:
            labels = [
                self._ytrain.coords[self._ytrain.dims[-1]].values[0].split("_", 1)[1],
                self._ytrain.coords[self._ytrain.dims[-1]].values[1].split("_", 1)[1],
            ]
            # Training plot
            disp = ConfusionMatrixDisplay(
                confusion_matrix(ytrain[..., -1], yhat_train[..., -1]),
                display_labels=labels,
            )
            disp.plot(ax=axs[0], values_format=".4g")
            disp.ax_.set_title("Train")
            disp.im_.colorbar.remove()

            # Testing plot
            disp = ConfusionMatrixDisplay(
                confusion_matrix(ytest[..., -1], yhat_test[..., -1]),
                display_labels=labels,
            )
            disp.plot(ax=axs[1], values_format=".4g")
            disp.ax_.set_ylabel("")
            disp.ax_.set_title("Test")
            disp.im_.colorbar.remove()

            plt.subplots_adjust(wspace=0.10, hspace=0.1)
            fig.colorbar(disp.im_, ax=axs)

        return fig, axs
