import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pyMAISE.explain.shap.explainers import (
    DeepExplainer,
    ExactExplainer,
    GradientExplainer,
    KernelExplainer,
)
from pyMAISE.explain.shap.plots._beeswarm import summary_legacy as summary_plot


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
    ax: matplotlib.pyplot.axis
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
    def __init__(
        self,
        base_model,
        X,
        feature_names=None,
        output_names=None,
        seed=None,
        **model_params,
    ):
        # Explainer Parameters
        self.model = base_model
        self.X = X
        self.shap_raw = {}
        self.shap_samples = {}
        self.feature_names = feature_names
        self.output_names = output_names
        if seed:
            np.random.seed(seed)

        self.n_features = self.X.shape[1]
        if self.feature_names is None:
            self.feature_names = np.array(
                ["FEATURE " + str(i) for i in range(self.n_features)]
            )

        # Infer number of outputs from the model prediction of two samples
        self.n_outputs = self.model.predict(self.X[0:2, :], verbose=0).shape[1]
        if self.output_names is not None:
            assert len(self.output_names) == self.n_outputs
        else:
            self.output_names = np.array(
                ["OUTPUT " + str(i) for i in range(self.n_outputs)]
            )

    def DeepLIFT(self, nsamples=None):
        """
        This functin fits a DeepLIFT explainer to evaluate SHAP coeffiicents (only for
        neural networks).

        Parameters
        ----------
        nsamples: int less than total samples in test set or None, default=None
            Number of samples used to estimate the DeepLIFT importances if it is
            different than using all samples in X
        """
        if nsamples is not None:
            test_indices = np.random.choice(
                self.X.shape[0], size=nsamples, replace=False
            )
            test_x_sample = self.X[test_indices]
        else:
            test_x_sample = self.X.copy()

        # Get the shap values for DeepLift using the sample Xtest set
        self.deep_lift = DeepExplainer(self.model, data=[self.X])
        deepshap_values = self.deep_lift.shap_values(test_x_sample)
        self.shap_raw["DeepLIFT"] = deepshap_values
        self.shap_samples["DeepLIFT"] = test_x_sample

    def IntGradients(self, nsamples=None):
        """
        This function fits an Integrated Gradient explainer to evaluate SHAP
        coeffiicents.

        Parameters
        ----------
        nsamples: int less than total samples in test set or None, default=None
            Number of test samples used to estimate the IG importances if it is
            different than using all samples in X
        """
        if nsamples is not None:
            test_indices = np.random.choice(
                self.X.shape[0], size=nsamples, replace=False
            )
            test_x_sample = self.X[test_indices]
        else:
            test_x_sample = self.X.copy()

        self.ig = GradientExplainer(self.model, data=[self.X])
        igshap_values = self.ig.shap_values(test_x_sample)

        self.shap_raw["IG"] = igshap_values
        self.shap_samples["IG"] = test_x_sample

    def KernelSHAP(self, n_background_samples=500, n_test_samples=200, n_bootstrap=200):
        """
        This function fits a Kernel SHAP explainer to evaluate SHAP coeffiicents.

        Parameters
        ----------
        n_background_samples: int less than total samples in X, default=500
            Number of training samples used as background for integrating out features
        n_test_samples: int less than total samples in X, default=200
            Number of
            test samples used to estimate the Kernel SHAP importances
        n_bootstrap: int, default=200
            Number of times to re-evaluate the model
            when explaining each prediction. More samples lead to lower variance
            estimates of the SHAP values.
        """
        if len(self.X) < n_background_samples:
            emsg = (
                "Total number of samples is less"
                "than requested number of background samples."
            )
            raise AttributeError(emsg)

        if len(self.X) < n_test_samples:
            emsg = (
                "Total number of samples is less than requested number of test samples."
            )
            raise AttributeError(emsg)

        indices = np.random.choice(
            self.X.shape[0], size=n_background_samples + n_test_samples, replace=False
        )
        background_data = self.X[indices[0:n_background_samples]]
        test_data = self.X[indices[n_background_samples:]]

        self.kernel_e = KernelExplainer(self.model, data=background_data)
        kernel_shap_values = self.kernel_e.shap_values(
            test_data, nsamples=n_bootstrap, silent=True
        )

        self.shap_raw["KernelSHAP"] = kernel_shap_values
        self.shap_samples["KernelSHAP"] = test_data

    def Exact_SHAP(self, nsamples=None):
        """
        This function fits an Exact SHAP explainer to evaluate SHAP coefficients.

        Parameters
        ----------
        nsamples: int less than total samples in X, default=None
            Number of test
            samples used to estimate the exact importances if it is different than using
            all samples in X
        """
        if nsamples is not None:
            test_indices = np.random.choice(
                self.X.shape[0], size=nsamples, replace=False
            )
            test_x_sample = self.X[test_indices]
        else:
            test_x_sample = self.X.copy()
        self.exact_e = ExactExplainer(self.model.predict, self.X)
        exact_shap_values = self.exact_e(test_x_sample).values
        if len(exact_shap_values.shape) != 3:
            self.shap_raw["ExactSHAP"] = exact_shap_values.reshape(
                exact_shap_values.shape[0], exact_shap_values.shape[1], 1
            )
        self.shap_raw["ExactSHAP"] = exact_shap_values
        self.shap_samples["ExactSHAP"] = test_x_sample

    def postprocess_results(self):
        self.shap_mean = {}
        self.shap_net_effect = {}
        for i, (key, value) in enumerate(self.shap_raw.items()):
            self.shap_mean[key] = pd.DataFrame(
                np.abs(self.shap_raw[key]).mean(axis=0),
                columns=self.output_names,
                index=self.feature_names,
            )

            # The total_effect is now a 2D array of features x outputs
            total_effect = self.shap_raw[key].sum(axis=0)
            # Normalize the values while preserving the sign
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
        Makes a beeswarm plot and bar plot for each shap method, or to make one for a
        particular method. If no output_index is given, make a plot for each.

        Parameters
        ----------
        output_name: str, default=None
            The name of the output variable the user
            is interested in plotting. Must be defined in output_names.
        output_index: int, default=None
            The index of the output variable the
            user is interested in plotting.
        method: str, default=None
            The key of the shap_raw array for the shap
            method a user wishes to plot. Options include: "DeepLIFT", "KernelSHAP",
            "IG", or "ExactSHAP"
        max_display: int, default=20
            The maximum number of input features that
            will be displayed on a beeswarm plot.
        run_name: str, default=None
            The name to save the figures as.
        save_figs: bool, default=True
            Whether or not to save the figures
            generated by this function.
        """
        if self.shap_mean is None or self.shap_net_effect is None:
            emsg = (
                "Results have not been post-processed. Please run"
                "post_process() method on your explain object prior to attempting"
                "plotting."
            )
            raise AttributeError(emsg)

        if output_name is not None and output_name not in self.output_names:
            emsg = (
                "The output you requested is not defined for this model."
                f"Valid output names include: {self.output_names}."
            )
            raise NameError(emsg)

        if output_index is None and output_name is not None:
            names = np.array(self.output_names)
            output_index = np.argwhere(names == output_name)[0][0]

        figsize = (18, 8)

        if method is None and output_index is None:
            output_indexes = [i for i in range(self.n_outputs)]
            methods = self.shap_raw.keys()

        elif method is None and output_index is not None:
            output_indexes = [output_index]
            methods = self.shap_raw.keys()

        elif method is not None and output_index is None:
            output_indexes = [i for i in range(self.n_outputs)]
            methods = [method]

        elif method is not None and output_index is not None:
            output_indexes = [output_index]
            methods = [method]

        for i in output_indexes:
            for key in methods:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
                fig.sca(ax1)
                summary_plot(
                    self.shap_raw[key][:, :, i],
                    features=self.shap_samples[key],
                    feature_names=self.feature_names,
                    show=False,
                    plot_size=None,
                    max_display=max_display,
                )
                ax1.set_title("Beeswarm Summary", loc="center")
                df_mean_sorted = (
                    self.shap_mean[key].iloc[:, i].sort_values(ascending=False)
                )
                df_neteffect_sorted = (
                    self.shap_net_effect[key].iloc[:, 0].loc[df_mean_sorted.index]
                )
                df_combined = pd.concat([df_mean_sorted, df_neteffect_sorted], axis=1)
                fig.sca(ax2)
                plot_bar_with_labels(df_combined, fig=fig, ax=ax2)
                fig.suptitle(
                    f"{key} {self.output_names[i]}",
                    fontsize="x-large",
                    fontweight="bold",
                    y=1.02,
                )
                fig.tight_layout()

                if save_figs:
                    if run_name is None:
                        fig.savefig(f"{key}_{i}.png", dpi=300)
                    else:
                        fig.savefig(f"{key}_{i}_{run_name}.png", dpi=300)
                else:
                    fig.show()

    def plot_bar_only(
        self,
        output_name=None,
        output_index=None,
        method=None,
        max_display=20,
        run_name=None,
        save_figs=True,
    ):
        """
        Makes a bar plot for each shap method, or to make one for a
        particular method. If no output_index is given, make a plot for each.

        Parameters
        ----------
        output_name: str, default=None
            The name of the output variable the user
            is interested in plotting. Must be defined in output_names.
        output_index: int, default=None
            The index of the output variable the
            user is interested in plotting.
        method: str, default=None
            The key of the shap_raw array for the shap
            method a user wishes to plot. Options include: "DeepLIFT", "KernelSHAP",
            "IG", or "ExactSHAP"
        max_display: int, default=20
            The maximum number of input features that
            will be displayed on a beeswarm plot.
        run_name: str, default=None
            The name to save the figures as.
        save_figs: bool, default=True
            Whether or not to save the figures
            generated by this function.
        """
        if self.shap_mean is None or self.shap_net_effect is None:
            emsg = (
                "Results have not been post-processed. Please run"
                "post_process() method on your explain object prior to attempting"
                "plotting."
            )
            raise AttributeError(emsg)

        if output_name is not None and output_name not in self.output_names:
            emsg = (
                "The output you requested is not defined for this model."
                f"Valid output names include: {self.output_names}."
            )
            raise NameError(emsg)

        if output_index is None and output_name is not None:
            names = np.array(self.output_names)
            output_index = np.argwhere(names == output_name)[0][0]

        if method is None and output_index is None:
            output_indexes = [i for i in range(self.n_outputs)]
            methods = self.shap_raw.keys()

        elif method is None and output_index is not None:
            output_indexes = [output_index]
            methods = self.shap_raw.keys()

        elif method is not None and output_index is None:
            output_indexes = [i for i in range(self.n_outputs)]
            methods = [method]

        elif method is not None and output_index is not None:
            output_indexes = [output_index]
            methods = [method]

        for i in output_indexes:
            for key in methods:
                fig, ax = plt.subplots()
                df_mean_sorted = (
                    self.shap_mean[key].iloc[:, i].sort_values(ascending=False)
                )
                df_neteffect_sorted = (
                    self.shap_net_effect[key].iloc[:, 0].loc[df_mean_sorted.index]
                )
                df_combined = pd.concat([df_mean_sorted, df_neteffect_sorted], axis=1)
                ax.bar(
                    df_combined.index,
                    df_combined.iloc[:, 0],
                    capsize=4,
                    width=0.3,
                    color="darkorchid",
                )
                ax.set_ylabel("Mean of |SHAP Values|")
                plt.xticks(rotation=90, ha="right")
                fig.tight_layout()

                if save_figs:
                    if run_name is None:
                        fig.savefig(f"{key}_{i}.png", dpi=300)
                    else:
                        fig.savefig(f"{key}_{i}_{run_name}.png", dpi=300)
                else:
                    fig.show()
