import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from captum.attr import DeepLiftShap, GradientShap, KernelShap, ShapleyValues


def plot_bar_with_labels(df, fig=None, ax=None):
    """
    Creates a bar plot showing mean of |SHAP values| for each feature.

    Parameters
    ----------
    df: pd.DataFrame
        A DataFrame with mean of absolute value
        of shap values for each feature in a model.
    Returns
    -------
    ax: matplotlib.pyplot.Axes
        The plot.
    """
    if fig is None or ax is None:
        fig, ax = plt.subplots()
    ax.bar(df.index, df.iloc[:, 0], capsize=4, width=0.3, color="darkorchid")

    ax.set_ylabel("Mean of |SHAP Values|")
    ax.set_title("Absolute Mean Importance", loc="center")
    plt.xticks(rotation=90, ha="right")
    plt.tight_layout()
    return ax


class ShapExplainers:
    """Explainers class based on Captum attributions.
    Allows for model-specific explainability features for
    a variety of methods, including DeepLIFT, KernelSHAP,
    and Integrated Gradients. Also features plotting capabilities
    for bar plots after attribution value
    calculations for any method.

    Parameters
    ----------
    base_model: model object.
        Must contain an associated .predict() method.
        Neural-network methods (DeepLIFT, IntGradients) additionally
        require a skorch NeuralNet with an accessible .module_ attribute.
    X: np.array.
        Array of feature values used for generating attribution values.
    feature_names: list, default=None.
        Ordered list of feature names corresponding to the columns
        in X for plotting.
    output_names: list, default=None.
        Ordered list of output names corresponding to those used
        to train base_model object for plotting.
    seed: int, default=None.
        Seed for reproducibility.
    """

    def __init__(
        self,
        base_model,
        X,
        feature_names=None,
        output_names=None,
        seed=None,
        **model_params,
    ):
        self.model = base_model
        self.X = np.asarray(X, dtype=np.float32)
        self.shap_raw = {}
        self.shap_samples = {}
        self.shap_mean = None
        self.shap_net_effect = None
        self.feature_names = feature_names
        self.output_names = output_names
        if seed:
            np.random.seed(seed)

        self.n_features = self.X.shape[1]
        if self.feature_names is None:
            self.feature_names = np.array(
                ["FEATURE " + str(i) for i in range(self.n_features)]
            )

        self.n_outputs = self.model.predict(self.X[0:2, :]).shape[1]
        if self.output_names is not None:
            assert len(self.output_names) == self.n_outputs
        else:
            self.output_names = np.array(
                ["OUTPUT " + str(i) for i in range(self.n_outputs)]
            )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _sample(self, nsamples):
        """Return (numpy array, float32 tensor) of nsamples rows from X (or all)."""
        if nsamples is not None:
            idx = np.random.choice(self.X.shape[0], size=nsamples, replace=False)
            arr = self.X[idx]
        else:
            arr = self.X.copy()
        return arr, torch.FloatTensor(arr)

    def _get_torch_module(self):
        """Return the nn.Module from a skorch wrapper (.module_), or None."""
        return getattr(self.model, "module_", None)

    def _make_forward_func(self):
        """Wrap model.predict() as a Tensor→Tensor callable for model-agnostic methods."""
        model = self.model

        def forward(x: torch.Tensor) -> torch.Tensor:
            return torch.FloatTensor(
                model.predict(x.detach().numpy().astype("float32"))
            )

        return forward

    def _run_attribution(
        self,
        key,
        captum_class,
        test_x,
        inputs,
        baselines,
        *,
        needs_module=True,
        per_sample=False,
        **attr_kwargs,
    ):
        """
        Generic attribution runner. Constructs a Captum explainer, calls
        .attribute() for each output target, stacks results into shape
        (n_samples, n_features, n_outputs), and stores in shap_raw/shap_samples.

        Parameters
        ----------
        key: str
            Key under which results are stored in shap_raw / shap_samples.
        captum_class: Captum attribution class
            Instantiated with the model argument (nn.Module or callable).
        test_x: np.array
            Test samples (numpy), stored in shap_samples.
        inputs: torch.Tensor
            Float32 tensor of test_x.
        baselines: torch.Tensor
            Reference/background tensor passed to .attribute().
        needs_module: bool, default=True
            True for gradient-based methods (DeepLiftShap, GradientShap) that
            require an nn.Module and support target= in .attribute().
            False for perturbation-based methods (KernelShap, ShapleyValues)
            that accept any callable. Gradient-based methods leave the module
            in eval() mode after attribution.
        per_sample: bool, default=False
            True for LIME-based methods (KernelShap) that must be called
            one sample at a time.
        **attr_kwargs
            Forwarded verbatim to explainer.attribute().
        """
        if needs_module:
            module = self._get_torch_module()
            if module is None:
                raise ValueError(
                    f"{captum_class.__name__} requires a PyTorch nn.Module "
                    "(skorch NeuralNet). Use KernelSHAP or Exact_SHAP for "
                    "classical sklearn models."
                )
            module.eval()
            explainer = captum_class(module)

            def _attr_one(x_in):
                return np.stack(
                    [
                        explainer.attribute(
                            x_in, baselines=baselines, target=i, **attr_kwargs
                        )
                        .detach()
                        .numpy()
                        for i in range(self.n_outputs)
                    ],
                    axis=2,
                )

        else:
            fwd = self._make_forward_func()
            explainer = captum_class(fwd)

            def _attr_one(x_in):
                return np.stack(
                    [
                        explainer.attribute(
                            x_in, baselines=baselines, target=i, **attr_kwargs
                        )
                        .detach()
                        .numpy()
                        for i in range(self.n_outputs)
                    ],
                    axis=2,
                )

        if per_sample:
            attributions = np.concatenate(
                [_attr_one(inputs[j : j + 1]) for j in range(inputs.shape[0])],
                axis=0,
            )
        else:
            attributions = _attr_one(inputs)

        self.shap_raw[key] = attributions
        self.shap_samples[key] = test_x

    # ------------------------------------------------------------------
    # Public attribution methods
    # ------------------------------------------------------------------

    def DeepLIFT(self, nsamples=None):
        """
        Fit a DeepLIFT (DeepLiftShap) explainer to evaluate attribution
        coefficients. Requires a skorch NeuralNet model.

        Parameters
        ----------
        nsamples: int or None, default=None.
            Number of test samples. Uses all X if None.
        """
        test_x, inputs = self._sample(nsamples)
        self._run_attribution(
            "DeepLIFT",
            DeepLiftShap,
            test_x,
            inputs,
            torch.FloatTensor(self.X),
            needs_module=True,
        )

    def IntGradients(self, nsamples=None):
        """
        Fit a GradientShap (Expected Gradients) explainer to evaluate
        attribution coefficients. Requires a skorch NeuralNet model.

        Parameters
        ----------
        nsamples: int or None, default=None.
            Number of test samples. Uses all X if None.
        """
        test_x, inputs = self._sample(nsamples)
        self._run_attribution(
            "IG",
            GradientShap,
            test_x,
            inputs,
            torch.FloatTensor(self.X),
            needs_module=True,
        )

    def KernelSHAP(self, n_background_samples=500, n_test_samples=200, n_bootstrap=200):
        """
        Fit a Kernel SHAP (KernelShap) explainer to evaluate attribution
        coefficients. Works with any model that has a .predict() method.

        Parameters
        ----------
        n_background_samples: int, default=500.
            Number of background samples used as baselines.
        n_test_samples: int, default=200.
            Number of test samples to explain.
        n_bootstrap: int, default=200.
            Number of perturbation samples per explanation.
        """
        if len(self.X) < n_background_samples:
            raise AttributeError(
                "Total number of samples is less"
                "than requested number of background samples."
            )
        if len(self.X) < n_test_samples:
            raise AttributeError(
                "Total number of samples is less than requested number of test samples."
            )

        indices = np.random.choice(
            self.X.shape[0],
            size=n_background_samples + n_test_samples,
            replace=False,
        )
        background_data = self.X[indices[:n_background_samples]]
        test_data = self.X[indices[n_background_samples:]]

        # KernelShap (LIME-based) requires a single baseline tensor; passing a
        # distribution of backgrounds batches them together and violates Captum's
        # scalar-output assertion. Use the mean background as the representative
        # baseline, which is standard practice for tabular KernelSHAP.
        mean_baseline = torch.FloatTensor(background_data.mean(axis=0, keepdims=True))

        self._run_attribution(
            "KernelSHAP",
            KernelShap,
            test_data,
            torch.FloatTensor(test_data),
            mean_baseline,
            needs_module=False,
            per_sample=True,
            n_samples=n_bootstrap,
            # perturbations_per_eval=1 keeps each masked-input batch at size 1 so
            # Captum's LIME assertion (numel(output) == len(inputs) == 1) holds.
            perturbations_per_eval=1,
        )

    def Exact_SHAP(self, nsamples=None):
        """
        Fit an exact Shapley value explainer. Works with any model that has a
        .predict() method. Only feasible for small feature counts (< ~10).

        Parameters
        ----------
        nsamples: int or None, default=None.
            Number of test samples. Uses all X if None.
        """
        test_x, inputs = self._sample(nsamples)
        self._run_attribution(
            "ExactSHAP",
            ShapleyValues,
            test_x,
            inputs,
            torch.zeros(1, self.n_features),
            needs_module=False,
        )

    # ------------------------------------------------------------------
    # Post-processing and plotting
    # ------------------------------------------------------------------

    def postprocess_results(self):
        self.shap_mean = {}
        self.shap_net_effect = {}
        for key in self.shap_raw:
            self.shap_mean[key] = pd.DataFrame(
                np.abs(self.shap_raw[key]).mean(axis=0),
                columns=self.output_names,
                index=self.feature_names,
            )

            total_effect = self.shap_raw[key].sum(axis=0)
            norm_effect = total_effect / np.sum(np.abs(total_effect), axis=0)
            self.shap_net_effect[key] = pd.DataFrame(
                norm_effect, columns=self.output_names, index=self.feature_names
            )

    def plot(
        self,
        output_name=None,
        output_index=None,
        method=None,
        max_display=20,
        run_name=None,
        save_figs=True,
    ):
        """
        Makes a bar plot for each attribution method (or a specific one).
        If no output_index is given, makes a plot for each output.

        Parameters
        ----------
        output_name: str, default=None.
            Name of the output to plot. Must be in output_names.
        output_index: int, default=None.
            Index of the output to plot.
        method: str, default=None.
            Key of the shap_raw array to plot. Options: "DeepLIFT",
            "KernelSHAP", "IG", "ExactSHAP".
        max_display: int, default=20.
            Maximum number of features to display.
        run_name: str, default=None.
            Filename prefix for saved figures.
        save_figs: bool, default=True.
            Whether to save figures to disk.
        """
        if self.shap_mean is None or self.shap_net_effect is None:
            raise AttributeError(
                "Results have not been post-processed. Please run"
                "post_process() method on your explain object prior to attempting"
                "plotting."
            )

        if output_name is not None and output_name not in self.output_names:
            raise NameError(
                "The output you requested is not defined for this model."
                f"Valid output names include: {self.output_names}."
            )

        if output_index is None and output_name is not None:
            names = np.array(self.output_names)
            output_index = np.argwhere(names == output_name)[0][0]

        if method is None and output_index is None:
            output_indexes = list(range(self.n_outputs))
            methods = self.shap_raw.keys()
        elif method is None and output_index is not None:
            output_indexes = [output_index]
            methods = self.shap_raw.keys()
        elif method is not None and output_index is None:
            output_indexes = list(range(self.n_outputs))
            methods = [method]
        else:
            output_indexes = [output_index]
            methods = [method]

        for i in output_indexes:
            for key in methods:
                fig, ax = plt.subplots()
                df_mean_sorted = (
                    self.shap_mean[key].iloc[:, i].sort_values(ascending=False)
                )
                if max_display is not None:
                    df_mean_sorted = df_mean_sorted.iloc[:max_display]
                df_neteffect_sorted = (
                    self.shap_net_effect[key].iloc[:, 0].loc[df_mean_sorted.index]
                )
                df_combined = pd.concat([df_mean_sorted, df_neteffect_sorted], axis=1)
                plot_bar_with_labels(df_combined, fig=fig, ax=ax)
                fig.suptitle(
                    f"{key} {self.output_names[i]}",
                    fontsize="x-large",
                    fontweight="bold",
                )
                fig.tight_layout()

                if save_figs:
                    fname = (
                        f"{key}_{i}.png"
                        if run_name is None
                        else f"{key}_{i}_{run_name}.png"
                    )
                    fig.savefig(fname, dpi=300)
                else:
                    fig.show()
