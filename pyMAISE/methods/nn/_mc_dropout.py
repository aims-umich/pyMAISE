import warnings
from typing import Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from skorch.history import History

from pyMAISE import settings
from pyMAISE.methods.nn._nn_hypermodel import nnHyperModel
from pyMAISE.methods.nn._utils import split_mean_var


class MCDropout:
    """
    Monte Carlo Dropout wrapper class around a single trained Skorch model.
    This user-facing class acts as a wrapper around a skorch NeuralNetRegressor
    to perform MC forward passes during inference for uncertainty quantification.
    """

    def __init__(
        self,
        model: Any,
        num_passes: int = 100,
        heteroscedastic: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        model: Any
            The underlying Skorch NeuralNetRegressor model.
        num_passes: int, default=100
            Number of Monte Carlo forward passes to perform during uncertainty estimation.
        heteroscedastic: bool, default=False
            Whether heteroscedastic loss (e.g., NLL) is used.
        """
        self.model = model
        self.num_passes = num_passes
        self.heteroscedastic = heteroscedastic

    def initialize(self) -> "MCDropout":
        """Initializes the underlying model."""
        self.model.initialize()
        return self

    @property
    def module_(self) -> Any:
        """Returns the PyTorch module of the underlying model."""
        return self.model.module_

    @property
    def history(self) -> History:
        """Returns the training history of the underlying model."""
        return self.model.history

    def fit(self, x: Any, y: Any, **kwargs) -> "MCDropout":
        """Fits the underlying model on the provided data."""
        self.model.fit(x, y, **kwargs)
        return self

    def predict(self, x: Any) -> np.ndarray:
        """
        **Deterministically** generate predictions for given input(s).
        Dropout is **not** applied.

        x: Any
            Input features.
        """
        raw = self.model.predict(x)
        if self.heteroscedastic:
            mean_t, _ = split_mean_var(torch.from_numpy(raw))
            return mean_t.numpy()
        return raw

    def predict_with_uncertainty(self,
            x: Any,
            num_passes: Optional[int] = None
    ) -> dict:
        """
        **Stochastically** generate predictions for given input(s).
        Dropout **is** applied.

        Parameters
        ----------
        x: Any
            Input features.
        num_passes: int or None, default=None
            Number of MC forward passes. If None, defaults to self.num_passes.

        Returns
        -------
        results: dict
            - predictions: array of shape (num_passes, n_samples, n_outputs)
            - mean: array of shape (n_samples, n_outputs)
            - epistemic_var: array of shape (n_samples, n_outputs) or (n_samples,)
            - aleatoric_var: array or None
        """
        n_passes = num_passes if num_passes is not None else self.num_passes
        module = self.model.module_

        # Collect all dropout submodules
        dropout_modules = [
            m for m in module.modules() if isinstance(m, nn.Dropout)
        ]

        raw_preds = []

        # Cannot just call model.predict() because skorch's predict()
        # turns dropout OFF, so this is the workaround:
        #   Convert inputs to a PyTorch float tensor on the same device as module
        device = next(module.parameters()).device
        if isinstance(x, torch.Tensor):
            x_tensor = x.to(dtype=torch.float32, device=device)
        elif x is not None:
            x_tensor = torch.tensor(np.asarray(x), dtype=torch.float32, device=device)
        else:
            x_tensor = None

        # Each pass has different dropout modules, so we turn them all OFF,
        # then turn ON for the specific pass we are predicting on.
        try:
            with torch.no_grad():
                for _ in range(n_passes):
                    module.eval()  # turns Dropout OFF
                    for m in dropout_modules:
                        m.train()  # turns Dropout ON

                    if x_tensor is not None:
                        # Direct PyTorch forward pass (bypassing skorch.predict)
                        # This is essentially the same as model.predict(X)
                        out = module(x_tensor).detach().cpu().numpy()
                    else:
                        warnings.warn("MCDropout: x is None. Falling back to model.predict(), which disables dropout sampling.")
                        out = self.model.predict(x)
                    raw_preds.append(out)
        finally:
            module.eval()

        predictions = np.stack(raw_preds, axis=0)

        if self.heteroscedastic:
            if predictions.shape[-1] % 2 != 0:
                raise ValueError("Heteroscedastic mode expects 2*n_targets outputs.")
            member_means_t, member_vars_t = split_mean_var(
                torch.from_numpy(predictions)
            )
            member_means = member_means_t.numpy()
            member_vars = member_vars_t.numpy()
            mean_preds = np.mean(member_means, axis=0)
            aleatoric_var = np.mean(member_vars, axis=0)

        else:
            mean_preds = np.mean(predictions, axis=0)
            aleatoric_var = None

        if settings.values.problem_type == settings.ProblemType.REGRESSION:
            if self.heteroscedastic:
                epistemic_var = np.var(member_means, axis=0)
            else:
                epistemic_var = np.var(predictions, axis=0)

        elif settings.values.problem_type == settings.ProblemType.CLASSIFICATION:
            epistemic_var = -np.sum(
                mean_preds * np.log(mean_preds + 1e-10), axis=-1
            )

        else:
            raise ValueError(
                "MCDropout.predict_with_uncertainty() only supports regression and classification problems."
            )

        return {
            "predictions": predictions,
            "mean": mean_preds,
            "epistemic_var": epistemic_var,
            "aleatoric_var": aleatoric_var,
        }


class MCDropoutHyperModel(nnHyperModel):
    """
    HyperModel for Monte Carlo Dropout extending pyMAISE's nnHyperModel.
    """

    def __init__(
        self,
        parameters: dict,
        input_shape: Tuple,
        name: str,
        num_passes: int = 100,
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
        num_passes: int, default=100
            Number of Monte Carlo forward passes to perform during uncertainty estimation.
        """
        super(MCDropoutHyperModel, self).__init__(parameters, input_shape, name)
        self.num_passes = num_passes
        self.best_trial = None

    def build(self, trial: Any) -> MCDropout:
        """
        Builds the underlying model and wraps it in an MCDropout instance.

        This method is primarily to be called internally by Optuna.

        Parameters
        ----------
        trial: Any
            Trial object from Optuna.

        Returns
        -------
        model: MCDropout
            The wrapped Monte Carlo Dropout model.
        """
        self.best_trial = trial
        if (
            self._compilation_params.get("loss") == "nll"
            and settings.values.problem_type == settings.ProblemType.CLASSIFICATION
        ):
            raise ValueError(
                "NLL loss is not currently supported for classification problems in MCDropoutHyperModel."
            )

        skorch_model = super(MCDropoutHyperModel, self).build(trial)
        return MCDropout(
            model=skorch_model,
            num_passes=self.num_passes,
            heteroscedastic=self._compilation_params.get("loss") == "nll",
        )
