import copy

from typing import Any, List, Optional, Tuple
import random
import numpy as np
import warnings

import torch
from skorch.history import History

from pyMAISE import settings
from pyMAISE.methods.nn._utils import split_mean_var
from pyMAISE.methods.nn._nn_hypermodel import nnHyperModel


"""
How the classes interact with pyMAISE:
1. Tuner: 
    - Instantiates a DeepEnsembleHyperModel.
    - Tunes a single, non-ensemble, model using the DeepEnsembleHyperModel object by calling the build method.
    - Returns the best hyperparameters.
2. PostProcessor:
    - Enables "Ensemble Mode" for the DeepEnsembleHyperModel.
    - When the PostProcessor invokes the fit method on the model, DeepEnsembleHyperModel returns the
      wrapper DeepEnsemble object encapsulating all the trained models.
3. User:
    - Getting the DeepEnsemble object from the PostProcessor by calling the get_model method allows the user
      to call predict_with_uncertainty, along with all the other typical methods associated with other pyMAISE models.
"""


class DeepEnsemble:
    """
    Deep Ensemble class for managing M independently trained models.
    Plain python class. This class acts as a wrapper around a list of nnHyperModels
    to work seamlessly in place of a skorch model.
    """

    def __init__(self, models: List, heteroscedastic: bool = False) -> None:
        """
        Parameters
        ----------
        models: List
            List of individual Skortch NeuralNetRegressor models.
        heteroscedastic: bool, default=False
            Whether to use heteroscedastic uncertainty quantification.

        Returns
        -------
        None
        """
        self.ensemble_models = models
        self.heteroscedastic = heteroscedastic
        self.history = History()

    def initialize(self) -> "DeepEnsemble":
        """
        Initializes each underlying model in the ensemble.
        """
        for model in self.ensemble_models:
            model.initialize()
        return self

    @property
    def module_(self) -> Any:
        """
        Returns the module structure of the first ensemble member.
        """
        return self.ensemble_models[0].module_

    def fit(self, x: Any, y: Any, **kwargs) -> "DeepEnsemble":
        """
        Trains each model in the ensemble to the provided data.

        Parameters
        ----------
        x: Any
            Input features/samples to fit on.
        y: Any
            Target values for the input features.
        **kwargs: dict
            Extra keyword arguments for model fitting.

        Returns
        -------
        self: DeepEnsemble
            The fitted DeepEnsemble instance.
        """
        for member in self.ensemble_models:
            member.fit(x, y, **kwargs)

        self._record_history()

        return self

    def predict(self, x: Any) -> np.ndarray:
        """
        Generates mean prediction across all ensemble members.

        For heteroscedastic (NLL) models, only the mean half of each
        member's output is used - the raw output is
        [mean | raw_variance] concatenated, and predict() returns
        values comparable to the target.

        Parameters
        ----------
        x: Any
            Input features/samples to predict on.

        Returns
        -------
        predictions: np.ndarray
            Mean prediction of shape (n_samples, n_targets), consistent
            with standard pyMAISE models. For full uncertainty
            decomposition use predict_with_uncertainty() instead.
        """
        stacked = self._predict_stacked(x)
        if self.heteroscedastic:
            mean_t, _ = split_mean_var(torch.from_numpy(stacked))
            stacked = mean_t.numpy()
        return np.mean(stacked, axis=0)

    def predict_with_uncertainty(self, x: Any) -> dict:
        """
        Generates predictions for the input samples using all models including variances.

        Aleatoric variance supported when loss ="nll" is set in compile_params.

        Parameters
        ----------
        x: Any
            Input features/samples.
        **kwargs: dict
            Extra keyword arguments for model prediction.

        Returns
        -------
        results: dict
            - **predictions** (np.ndarray, array-like): Stacked raw predictions from
              each ensemble member.

              Shapes:
                - Regression: ``(n_models, n_samples, n_outputs)``
                - Classification: ``(n_models, n_samples, n_classes)``

            - **mean_preds** (np.ndarray, array-like): Ensemble mean predictions.

              Shapes:
                - Regression: ``(n_samples, n_outputs)`` (average predicted target values)
                - Classification: ``(n_samples, n_classes)`` (average predicted class probability distributions)

            - **epistemic_var** (np.ndarray, array-like): Epistemic uncertainty.

              Shapes:
                - Regression: ``(n_samples, n_outputs)`` (variance of predictions across ensemble models)
                - Classification: ``(n_samples,)`` (predictive entropy over the class probabilities, computed as ``-sum(p * log(p))``)

            - **aleatoric_var** (np.ndarray or None): Aleatoric uncertainty.
            Returns None if heteroscedastic=False (default). Returns aleatoric
            variance if heteroscedastic=True (loss ="nll" in compile_params).

            Shapes:
                - Regression (heteroscedastic=True): ``(n_samples, n_outputs)``
                - Regression (heteroscedastic=False): ``None``
                - Classification: ``None`` (NLL not supported for classification in current implementation)
        """
        predictions = self._predict_stacked(x)

        if self.heteroscedastic:
            assert predictions.shape[-1] % 2 == 0, "heteroscedastic mode expects 2*n_targets outputs"
            # Reuse the same split + softplus transform used during training
            # (split_mean_var) so train and predict never disagree on what "variance" means.
            member_means_t, member_vars_t = split_mean_var(torch.from_numpy(predictions))
            member_means = member_means_t.numpy()
            member_vars = member_vars_t.numpy()
            mean_preds = np.mean(member_means, axis=0)
            aleatoric_var = np.mean(member_vars, axis=0)
        else:
            mean_preds = np.mean(predictions, axis=0)
            aleatoric_var = None


        if settings.values.problem_type == settings.ProblemType.REGRESSION:
            if self.heteroscedastic:
                # For heteroscedastic regression, epistemic variance is the variance of the mean predictions across ensemble members.
                epistemic_var = np.var(member_means, axis=0)
            else:
                epistemic_var = np.var(predictions, axis=0)
        elif settings.values.problem_type == settings.ProblemType.CLASSIFICATION:
            # For classification, use predictive entropy as the uncertainty measure.
            # entropy = -sum(p * log(p)), higher entropy = more uncertain
            epistemic_var = -np.sum(mean_preds * np.log(mean_preds + 1e-10), axis=-1)
        else:
            raise ValueError("DeepEnsemble.predict_with_uncertainty() only supports regression and classification problems.")
        
        return {
            "predictions": predictions,
            "mean": mean_preds,
            "epistemic_var": epistemic_var,
            "aleatoric_var": aleatoric_var
        }

    def _record_history(self):
        """
        Helper method to update the history object by averaging the history of all models.
        """
        all_history = [member.history for member in self.ensemble_models]

        if not all_history:
            return

        self.history = History()  # reset
        n_epoch = len(all_history[0])

        for i in range(n_epoch):
            self.history.new_epoch()

            # Training loss
            losses = [hist[i, "train_loss"] for hist in all_history]
            self.history.record("train_loss", np.mean(losses))
            self.history.record("train_loss_std", np.std(losses))

            # Validation loss
            if "valid_loss" in all_history[0][i]:
                val_losses = [hist[i, "valid_loss"] for hist in all_history]
                self.history.record("valid_loss", np.mean(val_losses))
                self.history.record("valid_loss_std", np.std(val_losses))

    def _predict_stacked(self, x: Any) -> np.ndarray:
        """
        Method returning stacked predictions from all ensemble members.

        Parameters
        ----------
        x: Any
            Input features/samples to predict on.

        Returns
        -------
        predictions: np.ndarray
            Stacked predictions of shape (n_models, n_samples, n_targets) in standard mode,
            or (n_models, n_samples, 2*n_targets) in heteroscedastic NLL mode.
            Used internally by predict_with_uncertainty. Not meant to be
            called directly by users.
        """
        return np.stack(
            [model.predict(x) for model in self.ensemble_models],
            axis=0,
        )


def _set_random_state(state: int):
    settings.values.random_state = state
    random.seed(state)
    np.random.seed(state)
    torch.manual_seed(state)
    torch.cuda.manual_seed_all(state)


class DeepEnsembleHyperModel(nnHyperModel):
    """
    HyperModel for Deep Ensembles extending pyMAISE's nnHyperModel.
    """

    def __init__(
        self,
        parameters: dict,
        input_shape: Tuple,
        name: str,
        num_models: int = 5,
        tune_ensemble: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        parameters: dict
            Hyperparameter space configuration options.
        input_shape: Tuple
            Shape of the input features.
        name: str
            Name identifier for this model.
        num_models: int, default=5
            Number of models in the ensemble.
        tune_ensemble: bool, default=False
            Whether to tune the ensemble as a whole.

        Returns
        -------
        None
        """
        super(DeepEnsembleHyperModel, self).__init__(parameters, input_shape, name)
        self.num_models = num_models
        self.ensemble_mode = tune_ensemble
        self.best_trial = None

    def build(self, trial: Any) -> Any:
        """
        Builds the model(s) for the Deep Ensemble.

        This method is primarily to be called internally by Optuna.

        Parameters
        ----------
        trial: Any
            Trial object from Optuna.

        Returns
        -------
        model: Model
            A Skorch NerualNetRegressor model (either DeepEnsembleWrapper or a single member).
        """
        self.best_trial = trial

        # Build each individual model and wrap them in a DeepEnsembleModel
        if self.ensemble_mode:
            return self.build_ensemble(trial)

        # Build the single model (best for normal tuning)
        return super(DeepEnsembleHyperModel, self).build(trial)


    def build_ensemble(self, trial: Any | None = None) -> DeepEnsemble:
        """
        Build each model in the ensemble, returning the wrapped ensemble model.

        This method is to be called by the user, and it can be called internally
        when `tune_ensemble=True`.

        Parameters
        ----------
        trial: Any
            HyperParameters object
                — If the model is returned from the Tuner, the best trial will be used from the
                Tuner. Otherwise, the trial parameter must be provided.

        Returns
        -------
        ensemble_model: DeepEnsemble
            The deep ensemble model.
        """
        # NLL loss not supported for classification
        if self._compilation_params.get("loss") == "nll" and settings.values.problem_type == settings.ProblemType.CLASSIFICATION:
            raise ValueError("NLL loss is not currently supported for classification problems in DeepEnsembleHyperModel.")
        
        models = []
        seed = settings.values.random_state

        # A random state must be set
        if seed is None:
            # Use numpy's random state acorss pyMAISE
            seed = np.random.randint(0, 2**32)

            # Set global random state
            warnings.warn(f"Required for building deep ensemble: Random State is None, setting to {seed}.")
            _set_random_state(seed)

        # Define which trial to use for all the ensemble members
        if trial is None and self.best_trial is None:
            raise ValueError("A trial parameter must be provided if the model has not been returned from the Tuner object.")
        this_trial = trial if trial is not None else self.best_trial

        for i in range(self.num_models):
            # Set the random seed across all frameworks
            _set_random_state(seed + i)

            # Generate the model based on the new seed
            models.append(super(DeepEnsembleHyperModel, self).build(this_trial))

        # reset the random state
        _set_random_state(seed)

        # Wrap them in the DeepEnsembleModel
        ensemble_model = DeepEnsemble(
            models=models,
            heteroscedastic=self._compilation_params.get("loss") == "nll"
        )
        return ensemble_model

    def fit(self, trial, model, x, y):
        """
        Overrides the fit method if the model is in ensemble mode to return
        the STD of the loss and validation loss.

        Returns
        -------
        history: History

        """
        history = super(DeepEnsembleHyperModel, self).fit(trial, model, x, y)
        if self.ensemble_mode:
            history["loss_std"] = list(model.history[:, "train_loss_std"])
            if len(model.history) > 0 and "valid_loss_std" in model.history[0]:
                history["val_loss_std"] = list(model.history[:, "valid_loss_std"])
        return history
