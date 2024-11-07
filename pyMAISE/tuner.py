import copy
import time

import keras_tuner as kt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras_tuner.oracles import (
    BayesianOptimizationOracle,
    GridSearchOracle,
    HyperbandOracle,
    RandomSearchOracle,
)
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from skopt import BayesSearchCV

import pyMAISE.settings as settings
from pyMAISE.methods import (
    SVM,
    DecisionTree,
    KNeighbors,
    LassoRegression,
    LinearRegression,
    LogisticRegression,
    NeuralNetsRegression,
    RandomForest,
    nnHyperModel,
    GaussianProcess,
    RidgeRegression,
    GradientBoosting,
    ElasticNet,
    AdaBoost,
    ExtraTrees,
    MultiOutput,
    Stacking,
)
from pyMAISE.utils import NNTuner, _try_clear


class Tuner:
    """
    Hyperparameter tuning object.

    .. _tuner_models:

    .. rubric:: Supported Models

    Supported models include

    - ``Linear``: linear `regressor <https://scikit-learn.org/stable/\
        modules/generated/sklearn.linear_model.LinearRegression.html>`_,
    - ``Lasso``: lasso `regressor <https://scikit-learn.org/stable\
        /modules/generated/sklearn.linear_model.Lasso.html>`_,
    - ``Logistic``: logistic `regressor <https://scikit-learn.org/stable\
        /modules/generated/sklearn.linear_model.LogisticRegression.html>`_,
    - ``SVM``: support vector `regressor <https://scikit-learn.org/stable\
        /modules/generated/sklearn.svm.SVR.html#sklearn.svm.SVR>`_
        and `classifier <https://scikit-learn.org/stable/modules/generated/\
        sklearn.svm.SVC.html#sklearn.svm.SVC>`_,
    - ``DT``: decision tree `regressor <https://scikit-learn.org/stable/\
        modules/generated/sklearn.tree.DecisionTreeRegressor.html>`_
        and `classifier <https://scikit-learn.org/stable/modules/generated/\
        sklearn.tree.DecisionTreeClassifier.html>`_,
    - ``RF``: random forest `regressor <https://scikit-learn.org/stable/\
        modules/generated/sklearn.ensemble.RandomForestRegressor.html>`_
        and `classifier <https://scikit-learn.org/stable/modules/generated/\
        sklearn.ensemble.RandomForestClassifier.html>`_,
    - ``KN``: k-nearest neighbors `regressor <https://scikit-learn.org/\
        stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html>`_
        and `classifier <https://scikit-learn.org/stable/modules/generated/\
        sklearn.neighbors.KNeighborsClassifier.html>`_,
    - ``EN``: elastic net `regressor <https://scikit-learn.org/stable/\
        modules/generated/sklearn.linear_model.ElasticNet.html>`_,
    - ``RD``: ridge `regressor <https://scikit-learn.org/stable/\
        modules/generated/sklearn.linear_model.Ridge.html>`_,
    - ``ET``: extra trees `regressor <https://scikit-learn.org/stable/\
        modules/generated/sklearn.ensemble.ExtraTreesRegressor.html>`_
        and `classifier <https://scikit-learn.org/stable/modules/generated/\
        sklearn.ensemble.ExtraTreesClassifier.html>`_,
    - ``AB``: AdaBoost `regressor <https://scikit-learn.org/stable/\
        modules/generated/sklearn.ensemble.AdaBoostRegressor.html>`_
        and `classifier <https://scikit-learn.org/stable/modules/generated/\
        sklearn.ensemble.AdaBoostClassifier.html>`_,
    - ``GP``: Gaussian process `regressor <https://scikit-learn.org/stable/\
        modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html>`_
        and `classifier <https://scikit-learn.org/stable/modules/generated/\
        sklearn.gaussian_process.GaussianProcessClassifier.html>`_,
    - ``GB``: gradient boosting `regressor <https://scikit-learn.org/stable/\
        modules/generated/sklearn.ensemble.GradientBoostingRegressor.html>`_
        and `classifier <https://scikit-learn.org/stable/modules/generated/\
        sklearn.ensemble.GradientBoostingClassifier.html>`_,
    - ``Stacking``: stacking `regressor <https://scikit-learn.org/stable/\
        modules/generated/sklearn.ensemble.StackingRegressor.html>`_
        and `classifier <https://scikit-learn.org/stable/modules/generated/\
        sklearn.ensemble.StackingClassifier.html>`_,
    - ``MultiOutput``: multioutput `regressor <https://scikit-learn.org/\
        stable/modules/generated/sklearn.multioutput.MultiOutputRegressor.html>`_
        and `classifier <https://scikit-learn.org/stable/modules/generated/\
        sklearn.multioutput.MultiOutputClassifier.html>`_.

    from :cite:`scikit-learn` and `sequential neural networks \
    <https://keras.io/guides/sequential_model/>`_ from :cite:`chollet2015keras`.

    .. _layersAndOptimizers:
    .. rubric:: Supported Neural Network Layers and Optimizers

    pyMAISE supports the following neural network layers using
    :cite:`chollet2015keras`:

    - ``Dense``: `dense <https://keras.io/api/layers/core_layers/dense/>`_,
    - ``Dropout``: `dropout <https://keras.io/api/layers/\
      regularization_layers/dropout/>`_,
    - ``LSTM``: `LSTM <https://keras.io/api/layers/recurrent_layers/lstm/>`_,
    - ``GRU``: `GRU <https://keras.io/api/layers/recurrent_layers/gru/>`_,
    - ``Conv1D``: `1D convolution <https://keras.io/api/layers/\
      convolution_layers/convolution1d/>`_,
    - ``Conv2D``: `2D convolution <https://keras.io/api/layers/\
      convolution_layers/convolution2d/>`_,
    - ``Conv3D``: `3D convolution <https://keras.io/api/layers/\
      convolution_layers/convolution3d/>`_,
    - ``MaxPooling1D``: `max pooling for 1D temporal data \
      <https://keras.io/api/layers/pooling_layers/max_pooling1d/>`_,
    - ``MaxPooling2D``: `max pooling for 2D temporal data \
      <https://keras.io/api/layers/pooling_layers/max_pooling2d/>`_,
    - ``MaxPooling3D``: `max pooling for 3D temporal data \
      <https://keras.io/api/layers/pooling_layers/max_pooling3d/>`_,
    - ``Flatten``: `flatten <https://keras.io/api/layers/\
      reshaping_layers/flatten/>`_,
    - ``Reshape``: `reshape <https://keras.io/api/layers/\
      reshaping_layers/reshape/>`_,

    and the following optimizers:

    - ``SGD``: `gradient descent <https://keras.io/api/optimizers/sgd/>`_,
    - ``RMSprop``: `RMSprop <https://keras.io/api/optimizers/rmsprop/>`_,
    - ``Adam``: `Adam <https://keras.io/api/optimizers/adam/>`_,
    - ``AdamW``: `AdamW <https://keras.io/api/optimizers/adamw/>`_,
    - ``Adadelta``: `Adadelta <https://keras.io/api/optimizers/adadelta/>`_,
    - ``Adagrad``: `Adagrad <https://keras.io/api/optimizers/adagrad/>`_,
    - ``Adamax``: `Adamax <https://keras.io/api/optimizers/adamax/>`_,
    - ``Adafactor``: `Adafactor <https://keras.io/api/optimizers/adafactor/>`_,
    - ``Nadam``: `Nadam <https://keras.io/api/optimizers/Nadam/>`_,
    - ``Ftrl``: `FTRL <https://keras.io/api/optimizers/ftrl/>`_.

    .. note:: For additional layer or optimizer support, submit a detailed issue at the
        `pyMAISE GitHub repository <https://github.com/myerspat/pyMAISE>`_ outlining the
        layer or optimizer required.

    Parameters
    ----------
    xtrain: xarray.DataArray
        Input training data.
    ytrain: xarray.DataArray
        Output training data.
    model_settings: dict of int, float, str, or pyMAISE.HyperParameters
        This dictionary specifies the name of the models of interest, which are assigned
        as a list to the ``models`` key. The model names are provided in the
        :ref:`tuner_models` section; all names that do not match those keys are assumed
        to be neural network models. For specific hyperparameters, please refer to the
        links provided for the models.

        For classical models, sklearn models :cite:`scikit-learn`, this
        dictionary specifies the hyperparameters different from default but
        remain constant throughout the hyperparameter tuning process. This is done by
        assigning a sub-dictionary under the key of the model's name.

        For neural network models :cite:`chollet2015keras`, this dictionary specifies
        both hyperparameters that remain constant throughout tuning and the tuning
        space using :class:`pyMAISE.Int`, :class:`pyMAISE.Float`,
        :class:`pyMAISE.Choice`, :class:`pyMAISE.Boolean`, and :class:`pyMAISE.Fixed`.
        This is done in the same way as classical models, where hyperparameters and
        their values are specified in sub-dictionaries under their model's key. In
        addition, number of layers, optimizers, wrappers, and sublayers can be
        specified.


    .. warning::
        When hyperparameter tuning a neural network with multiple of the same layer
        in one model, ensure the names of the layers are different, but the keywords are
        still present. For example, a dense sequential neural network with multiple
        dense layers can use names like ``Dense_input``, ``Dense_hidden``, and
        ``Dense_output``.

    Examples
    --------

    Given 2D input and output training data (``xtrain``, ``ytrain``) an example using
    linear and random forest models.

    .. code-block:: python

        import pyMAISE as mai

        model_settings = {
            "models": ["Linear", "RF"],
            "RF": {
                "n_estimators": 150,
            },
        }
        tuner = mai.Tuner(xtrain, ytrain, model_settings)

    From the above, we see we specify a linear model with default hyperparameters and
    a random forest model with all default hyperparameters except for 150 estimators.

    Given 3D input and 2D output time series data (``xtrain``, ``ytrain``) from
    :class:`pyMAISE.preprocessing.SplitSequence`, we can define a CNN-LSTM.

    .. code-block:: python

        import pyMAISE as mai
        from tensorflow.keras.layers import TimeDistributed
        from tensorflow.keras.callbacks import ReduceLROnPlateau

        cnn_lstm_structure = {
            "Reshape_input": {
                "target_shape": (2, 2, xtrain.shape[-1])
            },
            "Conv1D": {
                "filters": mai.Int(min_value=50, max_value=150),
                "kernel_size": 1,
                "activation": "relu",
                "wrapper": (
                    TimeDistributed, {
                        "input_shape": (None, 2, xtrain.shape[-1])
                    },
                ),
            },
            "MaxPooling1D": {
                "pool_size": 2,
                "wrapper": TimeDistributed,
            },
            "Flatten": {
                "wrapper": TimeDistributed,
            },
            "LSTM": {
                "num_layers": mai.Int(min_value=0, max_value=3),
                "units": mai.Int(min_value=20, max_value=100),
                "activation": "tanh",
                "recurrent_activation": "sigmoid",
                "recurrent_dropout": mai.Choice([0.0, 0.2, 0.4, 0.6]),
                "return_sequences": True,
            },
            "LSTM_output": {
                "units": mai.Int(min_value=20, max_value=100),
                "activation": "tanh",
                "recurrent_activation": "sigmoid",
            },
            "Dense": {
                "units": ytrain.shape[-1],
                "activation": "linear",
            },
        }
        fitting = {
            "batch_size": 512,
            "epochs": 5,
            "validation_split":0.15,
            "callbacks": [
                ReduceLROnPlateau(
                    monitor='val_mean_absolute_error',
                    factor=0.8,
                    patience=2,
                    min_lr=0,
                    verbose=1,
                ),
                EarlyStopping(
                    monitor="val_mean_absolute_error",
                    patience=3,
                )
            ]
        }
        adam = {
            "learning_rate": mai.Float(min_value=0.00001, max_value=0.001),
            "clipnorm": mai.Float(min_value=0.8, max_value=1.2),
            "clipvalue": mai.Float(min_value=0.3, max_value=0.7),
        }
        compiling = {
            "loss": "mean_absolute_error",
            "metrics": ["mean_absolute_error"],
        }

        model_settings = {
            "models": ["CNN-LSTM"],
            "CNN-LSTM": {
                "structural_params": cnn_lstm_structure,
                "optimizer": "Adam",
                "Adam": adam,
                "compile_params": compiling,
                "fitting_params": fitting,
            },
        }
        tuner = mai.Tuner(xtrain, ytrain, model_settings=model_settings)

    We see that we defined a neural network with 7 layers with the following
    tuning space:

    - 1D convolutional layer filters,
    - hidden LSTM number of layers,
    - hidden LSTM units,
    - hidden LSTM recurrent dropout,
    - output LSTM units,
    - Adam learning rate,
    - Adam clipnorm,
    - Adam clipvalue.

    Additionally, the ``Conv1D``, ``MaxPooling1D``, and ``Flatten`` layers use the
    ``keras.layers.TimeDistributed`` wrapper to accommodate the temporal dimension.
    """

    #: dict of pyMAISE.methods: Dictionary of supported models.
    supported_classical_models = {
        "Linear": LinearRegression,
        "Lasso": LassoRegression,
        "Logistic": LogisticRegression,
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
        "MultiOutput": MultiOutput,
        "Stacking": Stacking,
    }

    def __init__(self, xtrain, ytrain, model_settings):
        self._xtrain = xtrain.values
        self._ytrain = ytrain.values

        # Tuning loss for convergence plots
        self._tuning = {}

        # Throw error for call to SVM with multi-output
        if "SVM" in model_settings["models"] and self._ytrain.shape[-1] > 1:
            raise RuntimeError("SVM does not support multi-output data sets")

        # Fill models dictionary
        self._models = {}
        for model in model_settings["models"]:
            # Pull provided parameters
            parameters = model_settings[model] if model in model_settings else None

            # Add model object to dictionary
            if model in self.supported_classical_models:
                self._models[model] = copy.deepcopy(
                    self.supported_classical_models[model]
                )(parameters=parameters)
            elif settings.values.new_nn_architecture:
                self._models[model] = copy.deepcopy(
                    nnHyperModel
                    if settings.values.new_nn_architecture
                    else NeuralNetsRegression
                )(parameters=parameters, input_shape=self._xtrain.shape[1:], name=model)
            else:
                self._models[model] = copy.deepcopy(NeuralNetsRegression)(
                    parameters=parameters
                )

    # ===========================================================
    # Methods
    def grid_search(
        self,
        param_spaces,
        models=None,
        scoring=None,
        n_jobs=None,
        refit=True,
        cv=None,
        pre_dispatch="2*n_jobs",
    ):
        """
        Grid search over hyperparameter space for classical models. This function
        uses `sklearn.model_selection.GridSearchCV <https://scikit-learn.org/\
        stable/modules/generated/sklearn.model_selection.GridSearchCV.html>`_
        :cite:`scikit-learn`.

        Parameters
        ----------
        param_spaces: dict of dict of list
            The parameters which will be tuned through an exhaustive search
            over every configuration of hyperparameter in each model dictionary. Each
            parameter is defined as a dictionary key and assigned a list.
        models: list of str or None, default=None
            A list of model names that were defined in the initialization of
            :class:`pyMAISE.Tuner`. If ``None`` then all classical models are
            subject to grid search.


        .. note::
            For information on ``scoring``, ``n_jobs``, ``refit``, ``cv``,
            and ``pre_dispatch`` refer to `sklearn's documentation <https://\
            scikit-learn.org/stable/modules/generated/sklearn.model_selection.\
            GridSearchCV.html>`_.

        Returns
        -------
        data: dict of tuple(pd.DataFrame, model object)
            The hyperparameters and models for the top
            ``pyMAISE.Settings.num_configs_saved`` for each model. If fewer
            configurations are provided than ``pyMAISE.Settings.num_configs_saved``
            then all are taken.
        """
        print("Hyperparameter tuning classical models with grid search")

        return self._run_search(
            spaces=param_spaces,
            search_method=GridSearchCV,
            search_kwargs={
                "scoring": scoring,
                "n_jobs": n_jobs,
                "refit": refit,
                "cv": cv,
                "verbose": settings.values.verbosity,
                "pre_dispatch": pre_dispatch,
            },
            models=models,
        )

    def random_search(
        self,
        param_spaces,
        models=None,
        scoring=None,
        n_iter=10,
        n_jobs=None,
        refit=True,
        cv=None,
        pre_dispatch="2*n_jobs",
    ):
        """
        Random search over hyperparameter space for classical models. This function
        uses `sklearn.model_selection.RandomizedSearchCV <https://scikit-learn.org/\
        stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html>`_
        :cite:`scikit-learn`.

        Parameters
        ----------
        param_spaces: dict of dict of list or distributions
            The parameters which will be tuned through a random search
            over every configuration of hyperparameter in each model dictionary. Each
            parameter is defined as a dictionary key and assigned a list or distribution
            with an ``rvs`` method.
        models: list of str or None, default=None
            A list of model names defined in the initialization of
            :class:`pyMAISE.Tuner`. If ``None`` then all classical models are subject
            to grid search.


        .. note::
            For information on ``scoring``, ``n_iter``, ``n_jobs``, ``refit``, ``cv``,
            and ``pre_dispatch`` refer to `sklearn's documentation <https://\
            scikit-learn.org/stable/modules/generated/sklearn.model_selection.\
            RandomizedSearchCV.html>`_.

        Returns
        -------
        data: dict of tuple(pd.DataFrame, model object)
            The hyperparameters and models for the top
            ``pyMAISE.Settings.num_configs_saved`` for each model. If fewer
            configurations are provided than ``pyMAISE.Settings.num_configs_saved``
            then all are taken.
        """
        print("Hyperparameter tuning classical models with random search")

        return self._run_search(
            spaces=param_spaces,
            search_method=RandomizedSearchCV,
            search_kwargs={
                "scoring": scoring,
                "n_iter": n_iter,
                "n_jobs": n_jobs,
                "refit": refit,
                "cv": cv,
                "verbose": settings.values.verbosity,
                "random_state": settings.values.random_state,
                "pre_dispatch": pre_dispatch,
            },
            models=models,
        )

    def bayesian_search(
        self,
        param_spaces,
        models=None,
        scoring=None,
        n_iter=50,
        optimizer_kwargs=None,
        fit_params=None,
        n_jobs=None,
        n_points=1,
        refit=True,
        cv=None,
        pre_dispatch="2*n_jobs",
    ):
        """
        Bayesian search over hyperparameter space for classical models. This function
        uses `skopt.BayesSearchCV <https://scikit-optimize.github.io/stable/modules/\
        generated/skopt.BayesSearchCV.html>`_ :cite:`skopt`.

        Parameters
        ----------
        param_spaces: dict of dict of skopt.space.Dimension instance
            The parameters which will be tuned through a Bayesian search
            over every configuration of hyperparameter in each model dictionary. Each
            parameter is defined using ``skopt.space.Dimension`` instances
            (`Real <https://scikit-optimize.github.io/stable/modules/generated/\
            skopt.space.space.Integer.html>`_, `Integer <https://scikit-optimize.\
            github.io/stable/modules/generated/skopt.space.space.Integer.html>`_,
            or `Categorical <https://scikit-optimize.github.io/stable/modules/\
            generated/skopt.space.space.Categorical.html>`_).
        models: list of str or None, default=None
            A list of model names defined in the initialization of
            :class:`pyMAISE.Tuner`. If ``None`` then all classical models are subject
            to Bayesian search.


        .. note::
            For information on ``scoring``, ``n_iter``, ``optimizer_kwargs``,
            ``fit_params``, ``n_jobs``, ``n_points``, ``refit``, ``cv``, and
            ``pre_dispatch`` refer to
            `skopt's documentation <https://scikit-optimize.github.io/stable/\
            modules/generated/skopt.BayesSearchCV.html>`_.

        Returns
        -------
        data: dict of tuple(pd.DataFrame, model object)
            The hyperparameters and models for the top
            ``pyMAISE.Settings.num_configs_saved`` for each model. If fewer
            configurations are provided than ``pyMAISE.Settings.num_configs_saved``
            then all are taken.
        """
        print("Hyperparameter tuning classical models with bayesian search")

        return self._run_search(
            spaces=param_spaces,
            search_method=BayesSearchCV,
            search_kwargs={
                "n_iter": n_iter,
                "optimizer_kwargs": optimizer_kwargs,
                "scoring": scoring,
                "fit_params": fit_params,
                "n_jobs": n_jobs,
                "n_points": n_points,
                "pre_dispatch": pre_dispatch,
                "cv": cv,
                "refit": refit,
                "verbose": settings.values.verbosity,
                "random_state": settings.values.random_state,
            },
            models=models,
        )

    def manual_search(self, models=None, model_settings=None):
        """
        Fit a single hyperparameter configuration.

        Parameters
        ----------
        models: list of str or None, default=None
            The names of the models to be fit using manual search. If ``None``
            then all the models specified in the initialization of the
            :class:`pyMAISE.Tuner` are fit.
        model_settings: dict of int, float, or str
            The model settings for the models which are sub-dictionaries under
            the model key. If ``None`` then the hyperparameter configurations
            specified in the initialization of the :class:`pyMAISE.Tuner` are
            used.

        Returns
        -------
        data: dict of tuple(pd.DataFrame, model object)
            The hyperparameters and models for each model type.
        """
        # Get model types if not provided
        if models is None:
            models = list(self._models.keys())

        # Reshape if there is one feature
        xtrain = self._xtrain if self._xtrain.shape[-1] > 1 else self._xtrain[..., 0]
        ytrain = self._ytrain if self._ytrain.shape[-1] > 1 else self._ytrain[..., 0]

        data = {}
        for model in models:
            print(f"Tuning {model}")

            # Run model
            estimator = self._models[model].regressor()
            if model_settings is not None and model in model_settings:
                estimator.set_params(model_settings)

            resulting_model = estimator.fit(xtrain, ytrain)

            # Save model hyperparameters and the model itself
            data[model] = (
                pd.DataFrame({"params": [resulting_model.get_params()]}),
                resulting_model,
            )

        _try_clear()
        return data

    def _run_search(self, spaces, search_method, search_kwargs, models=None):
        if models is None:
            models = list(self._models.keys())
        models = [
            model
            for model in models
            if self.supported_classical_models.__contains__(model)
            or settings.values.new_nn_architecture is False
        ]

        # Reshape if there is one feature
        xtrain = self._xtrain if self._xtrain.shape[-1] > 1 else self._xtrain[..., 0]
        ytrain = self._ytrain if self._ytrain.shape[-1] > 1 else self._ytrain[..., 0]

        search_data = {}
        for model in models:
            if model in spaces:
                print(f"  Tuning {model}")

                # Run search method
                search = search_method(
                    self._models[model].regressor(), spaces[model], **search_kwargs
                )
                resulting_models = search.fit(xtrain, ytrain)

                # Save tuning results
                cv_results = pd.DataFrame(resulting_models.cv_results_)
                self._tuning[model] = np.array(
                    [
                        cv_results["mean_test_score"],
                        cv_results["std_test_score"],
                    ]
                )

                # Place parameter configurations in DataFrame and sort based on rank,
                # save the top num_configs_saved to the data dictionary
                top_configs = pd.DataFrame(
                    cv_results.sort_values("rank_test_score")["params"]
                )

                search_data[model] = (
                    top_configs[: settings.values.num_configs_saved],
                    resulting_models.best_estimator_,
                )

            else:
                print(
                    f"  Search space was not provided for {model}, "
                    + "current parameters will be added"
                )
                estimator = self._models[model].regressor()
                search_data = {
                    **search_data,
                    **{
                        model: (
                            pd.DataFrame({"params": [estimator.get_params()]}),
                            estimator,
                        )
                    },
                }

        _try_clear()
        return search_data

    def nn_grid_search(
        self,
        models=None,
        objective=None,
        max_trials=None,
        hyperparameters=None,
        allow_new_entries=True,
        tune_new_entries=True,
        max_retries_per_trial=0,
        max_consecutive_failed_trials=1,
        overwrite=True,
        directory="./",
        project_name="best_hp",
        cv=5,
        shuffle=False,
    ):
        """
        Grid search for neural networks. This function uses
        `keras_tuner.oracles.GridSearchOracle <https://keras.io/api/keras_tuner/\
        oracles/grid/>`_ with :class:`pyMAISE.CVTuner` for cross validation.
        Iterate over the defined search space and return the top models for each
        model type.

        Parameters
        ----------
        models: list of string or None, default=None
            The names of the neural network models for grid search. If ``None``, then
            all neural networks are fit with grid search.
        objective: str or keras_tuner.Objective, default=None
            The objective of the search. If the objective is a ``str`` of a
            sklearn.metrics, then that is used as the objective. Otherwise, the
            built-in objectives within KerasTuner are used.
        cv: int or cross-validation generator, default=5
            If an ``int``, then either
            `sklearn.model_selection.StratifiedKFold <https://scikit-learn.org/\
            stable/modules/generated/sklearn.model_selection.StratifiedKFold.html>`_
            or `sklearn.model_selection.KFold <https://scikit-learn.org/stable/\
            modules/generated/sklearn.model_selection.KFold.html>`_ are used depending
            on the ``pyMAISE.Settings.problem_type`` and output data type. If the
            problem is a classification problem and the output data is either binary or
            multiclass, then sklearn.model_selection.StratifiedKFold is used.
        shuffle: bool, default=False
            Whether to shuffle the data before cross-validation split.


        .. note::
            For information on ``max_trials``, ``hyperparameters``,
            ``allow_new_entries``,
            ``tune_new_entries``, ``max_consecutive_failed_trials``, ``overwrite``,
            ``directory``, and ``project_name`` refer to the
            `KerasTuner documentation \
            <https://keras.io/api/keras_tuner/oracles/grid/>`_.

        Returns
        -------
        data: dict of tuple(pd.DataFrame, model object)
            The hyperparameters and models for the top
            ``pyMAISE.Settings.num_configs_saved``
            for each model. If fewer configurations are provided, than
            ``pyMAISE.Settings.num_configs_saved`` then all are taken.
        """
        print("Hyperparameter tuning neural networks with grid search")

        kt_objective = self._determine_kt_objective(objective)
        oracle = GridSearchOracle(
            objective=kt_objective[0],
            max_trials=max_trials,
            seed=settings.values.random_state,
            hyperparameters=hyperparameters,
            tune_new_entries=tune_new_entries,
            allow_new_entries=allow_new_entries,
            max_retries_per_trial=max_retries_per_trial,
            max_consecutive_failed_trials=max_consecutive_failed_trials,
        )
        return self._nn_tuning(
            models=models,
            objective=objective,
            cv=cv,
            shuffle=shuffle,
            oracle=oracle,
            metrics=kt_objective[1],
            overwrite=overwrite,
            directory=directory,
            project_name=project_name,
        )

    def nn_random_search(
        self,
        models=None,
        objective=None,
        max_trials=10,
        hyperparameters=None,
        allow_new_entries=True,
        tune_new_entries=True,
        max_retries_per_trial=0,
        max_consecutive_failed_trials=1,
        overwrite=True,
        directory="./",
        project_name="best_hp",
        cv=5,
        shuffle=False,
    ):
        """
        Random search for neural networks. This function uses
        `keras_tuner.oracles.RandomSearchOracle <https://keras.io/api/\
        keras_tuner/oracles/random/>`_ with :class:`pyMIASE.CVTuner` for cross
        validation. Sample the defined search space based on a random distribution
        for each model type.

        Parameters
        ----------
        models: list of string or None, default=None
            The names of the neural network models for random search. If ``None``, then
            all neural networks are fit with random search.
        objective: str or keras_tuner.Objective, default=None
            The objective of the search. If the objective is a ``str`` of a
            sklearn.metrics, then that is used as the objective. Otherwise, the
            built-in objectives within KerasTuner are used.
        cv: int or cross-validation generator, default=5
            If an ``int``, then either
            `sklearn.model_selection.StratifiedKFold <https://scikit-learn.org/\
            stable/modules/generated/sklearn.model_selection.StratifiedKFold.html>`_
            or `sklearn.model_selection.KFold <https://scikit-learn.org/stable/\
            modules/generated/sklearn.model_selection.KFold.html>`_ are used depending
            on the ``pyMAISE.Settings.problem_type`` and output data type. If the
            problem is a classification problem and the output data is either binary or
            multiclass, then sklearn.model_selection.StratifiedKFold is used.
        shuffle: bool, default=False
            Whether to shuffle the data before cross-validation split.


        .. note::
            For information on ``max_trials``, ``hyperparameters``,
            ``allow_new_entries``,
            ``tune_new_entries``, ``max_retries_per_trial``,
            ``max_consecutive_failed_trials``,
            ``overwrite``, ``directory``, and ``project_name`` refer to `KerasTuner\
             documentation <https://keras.io/api/keras_tuner/oracles/random/>`_.

        Returns
        -------
        data: dict of tuple(pd.DataFrame, model object)
            The hyperparameters and models for the top
            ``pyMAISE.Settings.num_configs_saved``
            for each model. If fewer configurations are provided, than
            ``pyMAISE.Settings.num_configs_saved`` then all are taken.
        """
        print("Hyperparameter tuning neural networks with random search")

        kt_objective = self._determine_kt_objective(objective)
        oracle = RandomSearchOracle(
            objective=kt_objective[0],
            max_trials=max_trials,
            seed=settings.values.random_state,
            hyperparameters=hyperparameters,
            tune_new_entries=tune_new_entries,
            allow_new_entries=allow_new_entries,
            max_retries_per_trial=max_retries_per_trial,
            max_consecutive_failed_trials=max_consecutive_failed_trials,
        )
        return self._nn_tuning(
            models=models,
            objective=objective,
            cv=cv,
            shuffle=shuffle,
            oracle=oracle,
            metrics=kt_objective[1],
            overwrite=overwrite,
            directory=directory,
            project_name=project_name,
        )

    def nn_bayesian_search(
        self,
        models=None,
        objective=None,
        max_trials=10,
        num_initial_points=None,
        alpha=0.0001,
        beta=2.6,
        hyperparameters=None,
        tune_new_entries=True,
        allow_new_entries=True,
        max_retries_per_trial=0,
        max_consecutive_failed_trials=1,
        overwrite=True,
        directory="./",
        project_name="best_hp",
        cv=5,
        shuffle=False,
    ):
        """
        Bayesian search for neural networks. This function uses
        `keras_tuner.oracles.BayesianOptimizationOracle \
        <https://keras.io/api/keras_tuner/oracles/bayesian/>`_ with
        :class:`pyMAISE.CVTuner`
        for cross-validation. Iterate over sampled hyperparameter space using
        Bayesian optimization and return the top models for each model type.

        Parameters
        ----------
        models: list of string or None, default=None
            The names of the neural network models for Bayesian search. If ``None``,
            then all neural networks are fit with Bayesian search.
        objective: str or keras_tuner.Objective, default=None
            The objective of the search. If the objective is a ``str`` of a
            sklearn.metrics, then that is used as the objective. Otherwise the
            builtin objectives within KerasTuner are used.
        cv: int or cross-validation generator, default=5
            If an ``int`` then either
            `sklearn.model_selection.StratifiedKFold <https://scikit-learn.org/\
            stable/modules/generated/sklearn.model_selection.StratifiedKFold.html>`_
            or `sklearn.model_selection.KFold <https://scikit-learn.org/stable/\
            modules/generated/sklearn.model_selection.KFold.html>`_
            are used depending on
            the ``pyMAISE.Settings.problem_type`` and output data type. If the problem
            is a classification problem and the output data is either binary or
            multiclass, then sklearn.model_selection.StratifiedKFold is used.
        shuffle: bool, default=False
            Whether to shuffle the data before cross-validation split.


        .. note::
            For information on ``max_trials``, ``num_initial_points``,
            ``alpha``, ``beta``,
            ``hyperparameters``, ``tune_new_entries``, ``max_retries_per_trial``,
            ``max_consecutive_failed_trials``, ``overwrite``, ``directory``,
            and ``project_name`` refer to `KerasTuner documentation
            <https://keras.io/api/keras_tuner/oracles/bayesian/>`_.

        Returns
        -------
        data: dict of tuple(pd.DataFrame, model object)
            The hyperparameters and models for the top
            ``pyMAISE.Settings.num_configs_saved``
            for each model. If fewer configurations are provided, than
            ``pyMAISE.Settings.num_configs_saved`` then all are taken.
        """
        print("Hyperparameter tuning neural networks with bayesian search")

        kt_objective = self._determine_kt_objective(objective)
        oracle = BayesianOptimizationOracle(
            objective=kt_objective[0],
            max_trials=max_trials,
            num_initial_points=num_initial_points,
            alpha=alpha,
            beta=beta,
            seed=settings.values.random_state,
            hyperparameters=hyperparameters,
            tune_new_entries=tune_new_entries,
            allow_new_entries=allow_new_entries,
            max_retries_per_trial=max_retries_per_trial,
            max_consecutive_failed_trials=max_consecutive_failed_trials,
        )
        return self._nn_tuning(
            models=models,
            objective=objective,
            cv=cv,
            shuffle=shuffle,
            oracle=oracle,
            metrics=kt_objective[1],
            overwrite=overwrite,
            directory=directory,
            project_name=project_name,
        )

    def nn_hyperband_search(
        self,
        models=None,
        objective=None,
        max_epochs=100,
        factor=3,
        hyperband_iterations=1,
        hyperparameters=None,
        tune_new_entries=True,
        allow_new_entries=True,
        max_retries_per_trial=0,
        max_consecutive_failed_trials=3,
        overwrite=True,
        directory="./",
        project_name="best_hp",
        cv=5,
        shuffle=False,
    ):
        """
        Hyperband search for neural networks. This function uses
        `keras_tuner.oracles.HyperbandOracle <https://keras.io/api/\
        keras_tuner/oracles/hyperband/#hyperbandoracle-class>`_ with
        :class:`pyMAISE.CVTuner`
        for cross validation.

        Parameters
        ----------
        models: list of string or None, default=None
            The names of the neural network models for grid search. If ``None``, then
            all neural networks are fit with grid search.
        objective: str or keras_tuner.Objective, default=None
            The objective of the search. If the objective is a ``str`` of a
            sklearn.metrics then that is used as the objective. Otherwise the
            builtin objectives within KerasTuner are used.
        cv: int or cross-validation generator, default=5
            If an ``int``, then either
            `sklearn.model_selection.StratifiedKFold <https://scikit-learn.org/\
            stable/modules/generated/sklearn.model_selection.StratifiedKFold.html>`_
            or `sklearn.model_selection.KFold <https://scikit-learn.org/stable/\
            modules/generated/sklearn.model_selection.KFold.html>`_ are used
            depending on
            the ``pyMAISE.Settings.problem_type`` and output data type. If the problem
            is a classification problem and the output data is either binary or
            multiclass then sklearn.model_selection.StratifiedKFold is used.
        shuffle: bool, default=False
            Whether to shuffle the data before cross-validation split.


        .. note::
            For information on ``max_epochs``, ``factor``, ``hyperband_iterations``,
            ``hyperparameters``, ``tune_new_entries``, ``allow_new_entries``,
            ``max_retries_per_trial``, ``max_consecutive_failed_trials``,
            ``overwrite``, ``directory``, and ``project_name`` refer to
            `KerasTuner documentation <https://keras.io/api/keras_tuner/oracles/\
            hyperband/#hyperbandoracle-class>`_.

        Returns
        -------
        data: dict of tuple(pd.DataFrame, model object)
            The hyperparameters and models for the top
            ``pyMAISE.Settings.num_configs_saved``
            for each model. If fewer configurations are provided, than
            ``pyMAISE.Settings.num_configs_saved`` then all are taken.
        """
        print("Hyperparameter tuning neural networks with hyperband search")

        kt_objective = self._determine_kt_objective(objective)
        oracle = HyperbandOracle(
            objective=kt_objective[0],
            max_epochs=max_epochs,
            factor=factor,
            hyperband_iterations=hyperband_iterations,
            seed=settings.values.random_state,
            hyperparameters=hyperparameters,
            tune_new_entries=tune_new_entries,
            allow_new_entries=allow_new_entries,
            max_retries_per_trial=max_retries_per_trial,
            max_consecutive_failed_trials=max_consecutive_failed_trials,
        )
        return self._nn_tuning(
            models=models,
            objective=objective,
            cv=cv,
            shuffle=shuffle,
            oracle=oracle,
            metrics=kt_objective[1],
            overwrite=overwrite,
            directory=directory,
            project_name=project_name,
        )

    def _nn_tuning(
        self,
        models,
        objective,
        cv,
        shuffle,
        oracle,
        metrics,
        overwrite,
        directory,
        project_name,
    ):
        # Find all NN models if none are given by user
        if models is None:
            models = [
                model
                for model in self._models.keys()
                if not self.supported_classical_models.__contains__(model)
            ]

        data = {}
        timing = {}

        for model in models:
            start_time = time.time()

            # Initialize keras-tuner tuner
            tuner = NNTuner(
                objective=objective,
                cv=cv,
                shuffle=shuffle,
                hypermodel=self._models[model],
                oracle=copy.deepcopy(oracle),
                metrics=metrics,
                overwrite=overwrite,
                directory=directory,
                project_name=project_name,
                verbose=settings.values.verbosity,
            )

            # Run search
            tuner.search(
                x=self._xtrain, y=self._ytrain, verbose=settings.values.verbosity
            )

            # Get best hyperparameters
            best_hps = tuner.get_best_hyperparameters(settings.values.num_configs_saved)
            top_configs = pd.DataFrame({"params": best_hps})

            # Save test scores
            self._tuning[model] = np.array(
                [
                    tuner.mean_test_score,
                    tuner.std_test_score,
                ]
            )
            timing[model] = time.time() - start_time

            data[model] = (top_configs, tuner.hypermodel)

        if settings.values.verbosity > 0:
            print("\nTop Configurations")
            for model, (top_configs, _) in data.items():
                print(
                    f"\n-- {model} | Training Time: "
                    + f"{time.strftime('%T', time.gmtime(timing[model]))}"
                )
                for param, value in top_configs.iloc[0, 0].values.items():
                    print(f"{param}: {value}")

        _try_clear()
        return data

    def _determine_kt_objective(self, objective):
        """Determine objective from sklearn and make it compatible with keras_tuner."""
        if objective in ["r2_score", "accuracy_score"]:
            return (
                kt.Objective(objective, direction="max"),
                eval(f"{objective}"),
            )
        elif objective in [
            "f1_score",
            "mean_absolute_error",
            "mean_squared_error",
            "precision_score",
            "recall_score",
        ]:
            return (
                kt.Objective(objective, direction="min"),
                eval(f"{objective}"),
            )
        else:
            return (objective, None)

    def convergence_plot(self, ax=None, model_types=None):
        """
        Create a convergence plot for search using
        :attr:`pyMAISE.Tuner.cv_performance_data`.

        Parameters
        ----------
        ax: matplotlib.pyplot.axis or None, default=None
            Axis object. If ``None`` then one is created.
        model_types: list of str or None, default=None
            List of model names to add to the convergence plot. If ``None`` then
            all are added.


        Returns
        -------
        ax: matplotlib.pyplot.axis or None, default=None
            Axis object.
        """
        # If no models are provided fit all
        if model_types is None:
            model_types = list(self._tuning.keys())
        elif isinstance(model_types, str):
            model_types = [model_types]

        # Make axis if not given one
        if ax is None:
            ax = plt.gca()

        # For each model assert the performance metrics are the same size
        assert_shape = self._tuning[model_types[0]].shape

        for model in model_types:
            assert assert_shape == self._tuning[model].shape
            x = np.arange(self._tuning[model][0].size)
            ax.plot(
                x,
                self._tuning[model][0, :],
                linestyle="-",
                marker="o",
                label=model,
            )
            ax.fill_between(
                x,
                self._tuning[model][0, :] - 2 * self._tuning[model][1, :],
                self._tuning[model][0, :] + 2 * self._tuning[model][1, :],
                alpha=0.4,
            )

        # Show legend if length of models is more than one
        if len(model_types) > 1:
            ax.legend()

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Mean Test Score")

        return ax

    # Getters
    @property
    def cv_performance_data(self):
        """
        : list of float: Cross-validation performance, mean and standard deviation
                         of the test score, for each model.
        """
        return self._tuning
