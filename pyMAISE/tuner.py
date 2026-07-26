import copy
import os
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from optuna.samplers import GridSampler, RandomSampler, TPESampler
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

from pyMAISE.methods.nn import DeepEnsemble

# scikit-optimize 0.9.0 uses np.int which was removed in numpy 1.24.
# Patch before importing skopt so users aren't hit by the AttributeError.
if not hasattr(np, "int"):
    np.int = int  # type: ignore[attr-defined]

from skopt import BayesSearchCV

import pyMAISE.settings as settings
from pyMAISE.methods import (
    SVM,
    DecisionTree,
    KNeighbors,
    LassoRegression,
    LinearRegression,
    LogisticRegression,
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

    from :cite:`scikit-learn` and sequential neural networks built with PyTorch
    :cite:`pytorch`.

    .. _layersAndOptimizers:
    .. rubric:: Supported Neural Network Layers and Optimizers

    pyMAISE supports the following neural network layers via PyTorch:

    - ``Dense``: `linear <https://pytorch.org/docs/stable/generated/\
      torch.nn.Linear.html>`_,
    - ``Dropout``: `dropout <https://pytorch.org/docs/stable/generated/\
      torch.nn.Dropout.html>`_,
    - ``LSTM``: `LSTM <https://pytorch.org/docs/stable/generated/\
      torch.nn.LSTM.html>`_,
    - ``GRU``: `GRU <https://pytorch.org/docs/stable/generated/\
      torch.nn.GRU.html>`_,
    - ``Conv1D``: `1D convolution <https://pytorch.org/docs/stable/generated/\
      torch.nn.Conv1d.html>`_,
    - ``Conv2D``: `2D convolution <https://pytorch.org/docs/stable/generated/\
      torch.nn.Conv2d.html>`_,
    - ``Conv3D``: `3D convolution <https://pytorch.org/docs/stable/generated/\
      torch.nn.Conv3d.html>`_,
    - ``MaxPooling1D``: `max pooling for 1D temporal data \
      <https://pytorch.org/docs/stable/generated/torch.nn.MaxPool1d.html>`_,
    - ``MaxPooling2D``: `max pooling for 2D spatial data \
      <https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html>`_,
    - ``MaxPooling3D``: `max pooling for 3D volumetric data \
      <https://pytorch.org/docs/stable/generated/torch.nn.MaxPool3d.html>`_,
    - ``Flatten``: `flatten <https://pytorch.org/docs/stable/generated/\
      torch.nn.Flatten.html>`_,
    - ``Reshape``: reshape to a fixed target shape,

    and the following optimizers:

    - ``SGD``: `gradient descent <https://pytorch.org/docs/stable/generated/\
      torch.optim.SGD.html>`_,
    - ``RMSprop``: `RMSprop <https://pytorch.org/docs/stable/generated/\
      torch.optim.RMSprop.html>`_,
    - ``Adam``: `Adam <https://pytorch.org/docs/stable/generated/\
      torch.optim.Adam.html>`_,
    - ``AdamW``: `AdamW <https://pytorch.org/docs/stable/generated/\
      torch.optim.AdamW.html>`_,
    - ``Adadelta``: `Adadelta <https://pytorch.org/docs/stable/generated/\
      torch.optim.Adadelta.html>`_,
    - ``Adagrad``: `Adagrad <https://pytorch.org/docs/stable/generated/\
      torch.optim.Adagrad.html>`_,
    - ``Adamax``: `Adamax <https://pytorch.org/docs/stable/generated/\
      torch.optim.Adamax.html>`_.

    .. note:: For additional layer or optimizer support, submit a detailed issue at the
        `pyMAISE GitHub repository <https://github.com/aims-umich/pyMAISE>`_ outlining the
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

        For neural network models, this dictionary specifies both hyperparameters
        that remain constant throughout tuning and the tuning space using
        :class:`pyMAISE.Int`, :class:`pyMAISE.Float`, :class:`pyMAISE.Choice`,
        :class:`pyMAISE.Boolean`, and :class:`pyMAISE.Fixed`. This is done in
        the same way as classical models, where hyperparameters and their values
        are specified in sub-dictionaries under their model's key. In addition,
        number of layers, optimizers, and sublayers can be specified.


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
    :class:`pyMAISE.preprocessing.SplitSequence`, we can define a stacked LSTM
    network.

    .. code-block:: python

        import pyMAISE as mai

        lstm_structure = {
            "LSTM": {
                "num_layers": mai.Int(min_value=1, max_value=3),
                "units": mai.Int(min_value=20, max_value=100),
                "dropout": mai.Choice([0.0, 0.2, 0.4]),
                "return_sequences": True,
            },
            "LSTM_output": {
                "units": mai.Int(min_value=20, max_value=100),
            },
            "Dense": {
                "units": ytrain.shape[-1],
                "activation": "linear",
            },
        }
        fitting = {
            "batch_size": 512,
            "epochs": 5,
            "validation_split": 0.15,
        }
        adam = {
            "learning_rate": mai.Float(min_value=0.00001, max_value=0.001),
            "clipnorm": mai.Float(min_value=0.8, max_value=1.2),
            "clipvalue": mai.Float(min_value=0.3, max_value=0.7),
        }
        compiling = {
            "loss": "mean_absolute_error",
        }

        model_settings = {
            "models": ["LSTM-Net"],
            "LSTM-Net": {
                "structural_params": lstm_structure,
                "optimizer": "Adam",
                "Adam": adam,
                "compile_params": compiling,
                "fitting_params": fitting,
            },
        }
        tuner = mai.Tuner(xtrain, ytrain, model_settings=model_settings)

    We see that we defined a stacked LSTM network with the following tuning space:

    - number of hidden LSTM layers,
    - hidden LSTM units,
    - hidden LSTM dropout,
    - output LSTM units,
    - Adam learning rate,
    - Adam clipnorm,
    - Adam clipvalue.

    The ``LSTM`` entry defines the hidden layers (``num_layers`` controls how many
    are stacked with ``return_sequences=True``), and ``LSTM_output`` defines the
    final LSTM that collapses the sequence before the ``Dense`` output layer.
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

            # TODO Possibly converge these into a list and simply to a single lif statement?
            # Deep Ensemble
            elif model == "DE":
                from pyMAISE.methods.nn import DeepEnsembleHyperModel

                num_models = 5
                if parameters and "num_models" in parameters:
                    parameters = copy.deepcopy(parameters)
                    num_models = parameters.pop("num_models")

                self._models[model] = DeepEnsembleHyperModel(
                    parameters=parameters,
                    input_shape=self._xtrain.shape[1:],
                    name=model,
                    num_models=num_models,
                )

            # MC Dropout
            elif model == "MCD":
                from pyMAISE.methods.nn import MCDropoutHyperModel

                num_passes = 100
                if parameters and "num_passes" in parameters:
                    parameters = copy.deepcopy(parameters)
                    num_passes = parameters.pop("num_passes")

                self._models[model] = MCDropoutHyperModel(
                    parameters=parameters,
                    input_shape=self._xtrain.shape[1:],
                    name=model,
                    num_passes=num_passes,
                )

            else:
                self._models[model] = copy.deepcopy(nnHyperModel)(
                    parameters=parameters,
                    input_shape=self._xtrain.shape[1:],
                    name=model,
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
        models = [model for model in models if model in self.supported_classical_models]

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

                # Classical models don't use GPU. When joblib spawns worker
                # processes with n_jobs > 1, they inherit CUDA_VISIBLE_DEVICES
                # and can fail to deserialize sklearn objects due to CUDA
                # context conflicts. Hide GPUs from workers for the duration
                # of the search, then restore.
                _n_jobs = search_kwargs.get("n_jobs", 1)
                _cuda_env = os.environ.get("CUDA_VISIBLE_DEVICES")
                _hide_gpu = (
                    _n_jobs not in (None, 1)
                    and _cuda_env is not None
                    and _cuda_env != "-1"
                )
                if _hide_gpu:
                    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
                try:
                    resulting_models = search.fit(xtrain, ytrain)
                finally:
                    if _hide_gpu:
                        os.environ["CUDA_VISIBLE_DEVICES"] = _cuda_env

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
        cv=5,
        shuffle=False,
    ):
        """
        Grid search for neural networks using Optuna's ``GridSampler``.

        Exhaustively evaluates every combination of hyperparameter values
        defined in the model's search space.  The number of trials is computed
        automatically from the search space so no ``max_trials`` argument is
        needed.

        Parameters
        ----------
        models: list of str or None, default=None
            Neural network model names to tune. ``None`` tunes all NN models.
        objective: str or None, default=None
            Name of an sklearn metrics function (e.g. ``"r2_score"``,
            ``"mean_squared_error"``). ``None`` uses the default scoring
            (MSE for regression, error rate for classification).
        cv: int or cross-validation generator, default=5
            Number of folds or a pre-configured sklearn CV splitter.
        shuffle: bool, default=False
            Whether to shuffle data before splitting.

        Returns
        -------
        data: dict of tuple(pd.DataFrame, nnHyperModel)
            Top ``pyMAISE.Settings.num_configs_saved`` configurations per model.
        """
        print("Hyperparameter tuning neural networks with grid search")

        direction, metrics = self._determine_objective(objective)

        def sampler_factory(hypermodel):
            space = hypermodel.get_search_space()
            return GridSampler(space, seed=settings.values.random_state)

        def n_trials_factory(hypermodel):
            space = hypermodel.get_search_space()
            n = 1
            for values in space.values():
                n *= len(values)
            return n

        return self._nn_tuning(
            models=models,
            direction=direction,
            cv=cv,
            shuffle=shuffle,
            sampler_factory=sampler_factory,
            n_trials_factory=n_trials_factory,
            metrics=metrics,
        )

    def nn_random_search(
        self,
        models=None,
        objective=None,
        n_trials=10,
        cv=5,
        shuffle=False,
    ):
        """
        Random search for neural networks using Optuna's ``RandomSampler``.

        Samples ``n_trials`` hyperparameter configurations uniformly at random
        from the search space.

        Parameters
        ----------
        models: list of str or None, default=None
            Neural network model names to tune. ``None`` tunes all NN models.
        objective: str or None, default=None
            Name of an sklearn metrics function (e.g. ``"r2_score"``,
            ``"mean_squared_error"``). ``None`` uses the default scoring.
        n_trials: int, default=10
            Number of random configurations to evaluate.
        cv: int or cross-validation generator, default=5
            Number of folds or a pre-configured sklearn CV splitter.
        shuffle: bool, default=False
            Whether to shuffle data before splitting.

        Returns
        -------
        data: dict of tuple(pd.DataFrame, nnHyperModel)
            Top ``pyMAISE.Settings.num_configs_saved`` configurations per model.
        """
        print("Hyperparameter tuning neural networks with random search")

        direction, metrics = self._determine_objective(objective)

        return self._nn_tuning(
            models=models,
            direction=direction,
            cv=cv,
            shuffle=shuffle,
            sampler_factory=lambda _: RandomSampler(seed=settings.values.random_state),
            n_trials_factory=lambda _: n_trials,
            metrics=metrics,
        )

    def nn_bayesian_search(
        self,
        models=None,
        objective=None,
        n_trials=10,
        n_startup_trials=10,
        cv=5,
        shuffle=False,
    ):
        """
        Bayesian optimization search for neural networks using Optuna's ``TPESampler``.

        Tree-structured Parzen Estimator (TPE) builds a probabilistic model of
        the objective function and samples configurations likely to improve it.
        It starts with ``n_startup_trials`` random evaluations to seed the model,
        then switches to guided sampling.

        Parameters
        ----------
        models: list of str or None, default=None
            Neural network model names to tune. ``None`` tunes all NN models.
        objective: str or None, default=None
            Name of an sklearn metrics function (e.g. ``"r2_score"``,
            ``"mean_squared_error"``). ``None`` uses the default scoring.
        n_trials: int, default=10
            Total number of configurations to evaluate.
        n_startup_trials: int, default=10
            Number of random trials before TPE begins guided sampling.
        cv: int or cross-validation generator, default=5
            Number of folds or a pre-configured sklearn CV splitter.
        shuffle: bool, default=False
            Whether to shuffle data before splitting.

        Returns
        -------
        data: dict of tuple(pd.DataFrame, nnHyperModel)
            Top ``pyMAISE.Settings.num_configs_saved`` configurations per model.
        """
        print("Hyperparameter tuning neural networks with Bayesian search (TPE)")

        direction, metrics = self._determine_objective(objective)

        return self._nn_tuning(
            models=models,
            direction=direction,
            cv=cv,
            shuffle=shuffle,
            sampler_factory=lambda _: TPESampler(
                n_startup_trials=n_startup_trials,
                seed=settings.values.random_state,
            ),
            n_trials_factory=lambda _: n_trials,
            metrics=metrics,
        )

    def nn_hyperband_search(
        self,
        models=None,
        objective=None,
        n_trials=10,
        cv=5,
        shuffle=False,
    ):
        """
        Hyperband-style search for neural networks.

        .. note::
            True Hyperband prunes unpromising trials mid-training by hooking into
            epoch-level reporting.  That requires skorch callback integration not yet
            implemented in this backend, so this method currently uses
            ``TPESampler`` (the same sampler as :meth:`nn_bayesian_search`) and
            evaluates each trial to completion.  It is provided for API compatibility
            and will be upgraded to full Hyperband pruning in a future release.

        Parameters
        ----------
        models: list of str or None, default=None
            Neural network model names to tune. ``None`` tunes all NN models.
        objective: str or None, default=None
            Name of an sklearn metrics function (e.g. ``"r2_score"``,
            ``"mean_squared_error"``). ``None`` uses the default scoring.
        n_trials: int, default=10
            Number of configurations to evaluate.
        cv: int or cross-validation generator, default=5
            Number of folds or a pre-configured sklearn CV splitter.
        shuffle: bool, default=False
            Whether to shuffle data before splitting.

        Returns
        -------
        data: dict of tuple(pd.DataFrame, nnHyperModel)
            Top ``pyMAISE.Settings.num_configs_saved`` configurations per model.
        """
        warnings.warn(
            "nn_hyperband_search currently uses TPESampler rather than true Hyperband "
            "pruning.  Use nn_bayesian_search for equivalent behaviour, or wait for "
            "a future release with full Hyperband support.",
            UserWarning,
            stacklevel=2,
        )
        print("Hyperparameter tuning neural networks with hyperband search (TPE)")

        direction, metrics = self._determine_objective(objective)

        return self._nn_tuning(
            models=models,
            direction=direction,
            cv=cv,
            shuffle=shuffle,
            sampler_factory=lambda _: TPESampler(seed=settings.values.random_state),
            n_trials_factory=lambda _: n_trials,
            metrics=metrics,
        )

    def _nn_tuning(
        self,
        models,
        direction,
        cv,
        shuffle,
        sampler_factory,
        n_trials_factory,
        metrics,
    ):
        """
        Inner driver for all four NN search methods.

        Parameters
        ----------
        sampler_factory: callable
            ``sampler_factory(hypermodel) -> optuna.samplers.BaseSampler``.
            Called per model so grid search can derive its sampler from the
            model's search space via ``hypermodel.get_search_space()``.
        n_trials_factory: callable
            ``n_trials_factory(hypermodel) -> int``.
            Called per model for the same reason.
        """
        # Find all NN models if none are given by user
        if models is None:
            models = [
                model
                for model in self._models.keys()
                if model not in self.supported_classical_models
            ]

        data = {}
        timing = {}

        for model in models:
            start_time = time.time()
            hypermodel = self._models[model]

            tuner = NNTuner(
                hypermodel=hypermodel,
                sampler=sampler_factory(hypermodel),
                n_trials=n_trials_factory(hypermodel),
                objective=model,
                cv=cv,
                shuffle=shuffle,
                metrics=metrics,
                direction=direction,
            )

            tuner.search(x=self._xtrain, y=self._ytrain)

            # Sort completed trials by objective value and keep the top configs.
            # Trials are plain dicts of {param_name: value}, so they serialise
            # directly into a DataFrame without any keras-tuner wrapper needed.
            reverse = direction == "maximize"
            best_trials = sorted(
                tuner.study.trials, key=lambda t: t.value, reverse=reverse
            )
            best_params = [
                t.params for t in best_trials[: settings.values.num_configs_saved]
            ]
            top_configs = pd.DataFrame({"params": best_params})

            self._tuning[model] = np.array(
                [tuner.mean_test_score, tuner.std_test_score]
            )
            timing[model] = time.time() - start_time

            data[model] = (top_configs, hypermodel)

        if settings.values.verbosity > 0:
            print("\nTop Configurations")
            for model, (top_configs, _) in data.items():
                print(
                    f"\n-- {model} | Training Time: "
                    + f"{time.strftime('%T', time.gmtime(timing[model]))}"
                )
                # top_configs.iloc[0, 0] is a plain dict; no .values wrapper needed
                for param, value in top_configs.iloc[0, 0].items():
                    print(f"{param}: {value}")

        _try_clear()
        return data

    def _determine_objective(self, objective):
        """
        Return ``(direction, metrics_callable)`` for the given objective name.

        ``direction`` is ``"maximize"`` or ``"minimize"`` and is passed directly
        to ``optuna.create_study``.  ``metrics_callable`` is the corresponding
        sklearn metrics function, or ``None`` when ``objective`` is ``None`` or an
        unrecognised string (NNTuner will use its built-in default in that case).
        """
        _maximize = {"r2_score", "accuracy_score"}
        _minimize = {
            "f1_score",
            "mean_absolute_error",
            "mean_squared_error",
            "precision_score",
            "recall_score",
        }

        if objective in _maximize:
            return "maximize", eval(objective)
        elif objective in _minimize:
            return "minimize", eval(objective)
        else:
            return "minimize", None

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
