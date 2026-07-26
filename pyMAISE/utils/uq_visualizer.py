import copy
import warnings

import numpy as np
import matplotlib.pyplot as plt


class UQVisualizer:
    """
    Visualizer for Uncertainty Quantification (UQ) models supporting
    predict_with_uncertainty(x) (Deep Ensembles, MC Dropout, Bayesian NNs).
    """

    def __init__(self, model, xtrain, xtest, ytrain, ytest, yscaler):
        self.model = model
        self.xtrain = xtrain
        self.xtest = xtest
        self.ytrain = ytrain
        self.ytest = ytest
        self.yscaler = yscaler

        # Target names mapped to indices for feature string mapping
        self.target_names = ytrain.coords[ytrain.dims[-1]].values

        # Compute and cache train/test uncertainty predictions
        if model is not None and hasattr(model, "predict_with_uncertainty"):
            self.test_uq_predictions = model.predict_with_uncertainty(self.xtest.values)
            self.train_uq_predictions = model.predict_with_uncertainty(self.xtrain.values)
            self.uq_predictions = self.test_uq_predictions
            self.has_aleatoric = self.uq_predictions.get("aleatoric_var") is not None
        else:
            self.test_uq_predictions = None
            self.train_uq_predictions = None
            self.uq_predictions = None
            self.has_aleatoric = False

    def sorted_uncertainty_plot(
        self,
        ax=None,
        model=None,
        feature: str | None = None,
        show_members: bool = False,
    ) -> plt.Axes:
        """
        Plot mean predictions sorted in ascending order along with epistemic
        and optional aleatoric uncertainty bands (+/- 1 sigma) for a specified
        target feature. Optionally overlay individual ensemble member predictions.

        Parameters
        ----------
        ax: matplotlib.pyplot.axis or None, default=None
            If not given, then an axis is created.
        model: Model wrapper or None, default=None
            Optional UQ model to compute uncertainty predictions from.
            If None or self.model, uses precomputed uncertainty predictions from initialization.
        feature: str or None, default=None
            The name of the target feature to plot. If None, defaults to the first
            target feature in target_names.
        show_members: bool, default=False
            If True, overlay individual ensemble member/pass predictions as dashed lines.

        Returns
        -------
        ax: matplotlib.pyplot.axis
            The plot axis.
        """
        ax = ax or plt.gca()

        # Map feature string to index
        f = feature or self.target_names[0]
        try:
            idx = list(self.target_names).index(f)
        except ValueError:
            warnings.warn(f"UQVisualizer: Feature {f} not found in target names. Skipping plot.")
            return ax

        # Use cached predictions if model is None or self.model
        if model is not None and model is not self.model:
            uq_predictions = model.predict_with_uncertainty(self.xtest.values)
            has_aleatoric = uq_predictions.get("aleatoric_var") is not None
        else:
            uq_predictions = self.test_uq_predictions
            has_aleatoric = self.has_aleatoric

        if uq_predictions is None:
            warnings.warn("UQVisualizer: No uncertainty predictions available. Skipping plot.")
            return ax

        epistemic = uq_predictions["epistemic_var"]
        if has_aleatoric:
            aleatoric = uq_predictions["aleatoric_var"]

        # Apply inverse transform before slicing
        if self.yscaler is not None:
            mean_all = self.yscaler.inverse_transform(uq_predictions["mean"])
            scale = self.yscaler.scale_[idx]
        else:
            mean_all = uq_predictions["mean"]
            scale = 1.0

        # Slice target feature
        mean = mean_all[:, idx]
        if epistemic.ndim > 1:
            epistemic = epistemic[:, idx]
        if has_aleatoric and aleatoric.ndim > 1:
            aleatoric = aleatoric[:, idx]

        # Sort predictions
        sorted_idxs = np.argsort(mean)
        x = np.arange(sorted_idxs.shape[0])
        mean_sorted = mean[sorted_idxs]

        # Calculate standard deviations
        if has_aleatoric:
            al_std_sorted = np.sqrt(np.maximum(0, aleatoric[sorted_idxs])) / scale
        ep_std_sorted = np.sqrt(np.maximum(0, epistemic[sorted_idxs])) / scale

        if has_aleatoric:
            ax.fill_between(
                x,
                mean_sorted - al_std_sorted,
                mean_sorted + al_std_sorted,
                alpha=0.2,
                color="r",
                label=r"Aleatoric $\sigma$",
            )
        ax.fill_between(
            x,
            mean_sorted - ep_std_sorted,
            mean_sorted + ep_std_sorted,
            alpha=0.2,
            color="g",
            label=r"Epistemic $\sigma$",
        )
        # Plot member trajectories
        if show_members and "predictions" in uq_predictions:
            raw_preds = uq_predictions["predictions"]
            if has_aleatoric:
                import torch
                from pyMAISE.methods.nn._utils import split_mean_var
                member_means_t, _ = split_mean_var(torch.from_numpy(raw_preds))
                member_means = member_means_t.numpy()
            else:
                member_means = raw_preds

            for m in range(member_means.shape[0]):
                if self.yscaler is not None:
                    m_mean_all = self.yscaler.inverse_transform(member_means[m])
                else:
                    m_mean_all = member_means[m]
                m_mean_sorted = m_mean_all[:, idx][sorted_idxs]
                ax.plot(
                    x,
                    m_mean_sorted,
                    linestyle="--",
                    alpha=0.6,
                    lw=1,
                    color="blue",
                    label="Member Prediction" if m == 0 else None,
                )

        ax.plot(x, mean_sorted, lw=2, c="k", label="Mean")
        ax.legend()
        ax.set_title(f"Feature: {f}")
        ax.set_ylabel("Prediction Value")
        ax.set_xlabel("Sample Index")
        return ax

    def epistemic_aleatoric_plot(
        self,
        ax=None,
        model=None,
        normalize: bool = False,
    ) -> plt.Axes:
        """
        Create a bar chart comparing epistemic and aleatoric variance per
        output variable for a UQ model.

        Parameters
        ----------
        ax: matplotlib.pyplot.axis or None, default=None
            If not given, then an axis is created.
        model: UQ model or None, default=None
            Optional UQ model to compute uncertainty predictions from.
            If None or self.model, uses cached predictions.
        normalize: bool, default=False
            If True, normalize both epistemic and aleatoric variance to [0, 1].

        Returns
        -------
        ax: matplotlib.pyplot.axis
            The plot axis.
        """
        if model is not None and model is not self.model:
            unc = model.predict_with_uncertainty(self.xtest.values)
            has_aleatoric = unc.get("aleatoric_var") is not None
        else:
            unc = self.test_uq_predictions
            has_aleatoric = self.has_aleatoric

        if unc is None:
            warnings.warn("UQVisualizer: No uncertainty predictions available. Skipping plot.")
            return ax or plt.gca()

        epistemic = np.mean(unc["epistemic_var"], axis=0)
        aleatoric = (
            np.mean(unc["aleatoric_var"], axis=0)
            if has_aleatoric else None
        )

        if normalize:
            all_vals = np.concatenate([epistemic, aleatoric]) if has_aleatoric else epistemic
            max_val = np.max(all_vals) if np.max(all_vals) > 0 else 1.0
            epistemic_norm = epistemic / max_val
            aleatoric_norm = aleatoric / max_val if has_aleatoric else None
            ylabel = "Normalized Variance (0-1)"
        else:
            epistemic_norm = epistemic
            aleatoric_norm = aleatoric
            ylabel = "Variance"

        if hasattr(self.ytest, "coords"):
            output_names = list(self.ytest.coords[self.ytest.dims[-1]].values)
        else:
            output_names = [f"Output {i}" for i in range(len(epistemic))]

        n_outputs = len(output_names)
        x = np.arange(n_outputs)
        width = 0.35

        if ax is None:
            ax = plt.gca()

        ax.bar(
            x - width / 2 if has_aleatoric else x,
            epistemic_norm,
            width,
            label=r"Epistemic $\sigma^2$",
            color="steelblue",
        )

        if has_aleatoric:
            ax.bar(
                x + width / 2,
                aleatoric_norm,
                width,
                label=r"Aleatoric $\sigma^2$",
                color="orange",
            )

        ax.set_xticks(x)
        ax.set_xticklabels(output_names, rotation=45, ha="right")
        ax.set_ylabel(ylabel)
        if normalize:
            ax.set_ylim(0, 1.05)
        ax.set_title("Epistemic vs Aleatoric Uncertainty per Output")
        ax.legend()

        return ax

    def data_calibration_plot(
        self,
        ax=None,
        model=None,
        sections: int = 6,
        plot_type: str = "sorted_uncertainty",
        feature: str | None = None,
        normalize: bool = False,
        **kwargs,
    ):
        """
        Create a grid of subplots showing how uncertainty calibration evolves
        as the training dataset size increases across incremental data sections.

        Parameters
        ----------
        ax: np.ndarray of matplotlib.pyplot.axis, matplotlib.pyplot.axis, or None, default=None
            Subplot axis array or parent axis. If None, a figure and subplot grid
            is automatically created based on sections.
        model: UQ model or None, default=None
            Optional UQ model to compute uncertainty predictions from.
            If None or self.model, uses self.model.
        sections: int, default=6
            The number of incremental data sections to split training data into.
        plot_type: str, default="sorted_uncertainty"
            The visualization plot type to render on each section subplot.
            Supported options: "sorted_uncertainty" ("su"), "epistemic_aleatoric" ("ea").
        feature: str or None, default=None
            Target feature name to visualize (for plot_type="sorted_uncertainty").
        normalize: bool, default=False
            Whether to normalize variances for plot_type="epistemic_aleatoric".

        Returns
        -------
        fig, axes: matplotlib.pyplot.Figure, np.ndarray of matplotlib.pyplot.axis
            The created or modified figure and subplot axes array.
        """
        if sections == 4:
            n_rows, n_cols = 2, 2
        elif sections == 6:
            n_rows, n_cols = 2, 3
        elif sections == 10:
            n_rows, n_cols = 2, 5
        else:
            n_cols = int(np.ceil(np.sqrt(sections)))
            n_rows = int(np.ceil(sections / n_cols))

        if ax is None:
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
        else:
            fig = None
            axes = ax

        axes_flat = np.array(axes).flatten() if isinstance(axes, (np.ndarray, list)) else [axes]

        f = feature or self.target_names[0]
        try:
            feature_idx = list(self.target_names).index(f)
        except ValueError:
            feature_idx = 0

        target_model = model if model is not None else self.model

        n_samples = self.xtrain.shape[0]
        step = max(1, n_samples // sections)

        for i in range(sections):
            if i >= len(axes_flat):
                break

            sub_ax = axes_flat[i]
            end_idx = n_samples if i == sections - 1 else (i + 1) * step
            pct = f"{int(round((end_idx / n_samples) * 100))}%"

            xtrain_sub = self.xtrain.values[:end_idx]
            ytrain_sub = self.ytrain.values[:end_idx]

            sub_ensemble = copy.deepcopy(target_model)
            sub_ensemble.fit(xtrain_sub, ytrain_sub)

            match plot_type.lower():
                case "sorted_uncertainty" | "su":
                    self.sorted_uncertainty_plot(ax=sub_ax, model=sub_ensemble, feature=f)
                case "epistemic_aleatoric" | "ea":
                    self.epistemic_aleatoric_plot(ax=sub_ax, model=sub_ensemble, normalize=normalize)
                case _:
                    raise ValueError(
                        f"Unknown plot_type: '{plot_type}'. "
                        f"Supported options: 'sorted_uncertainty' ('su'), 'epistemic_aleatoric' ('ea')."
                    )

            sub_ax.set_title(f"Data: {pct} ({end_idx} samples)")

            if kwargs.get("show_stats", True):
                unc_sub = sub_ensemble.predict_with_uncertainty(self.xtest.values)
                ep_v = unc_sub["epistemic_var"]
                if ep_v.ndim > 1:
                    ep_v = ep_v[:, feature_idx] if plot_type.lower() in ["sorted_uncertainty", "su"] else np.mean(ep_v, axis=-1)

                scale = self.yscaler.scale_[feature_idx] if (self.yscaler is not None and hasattr(self.yscaler, "scale_")) else 1.0
                ep_std = np.sqrt(np.maximum(0, ep_v)) / scale
                mean_ep = np.mean(ep_std)

                if unc_sub.get("aleatoric_var") is not None:
                    al_v = unc_sub["aleatoric_var"]
                    if al_v.ndim > 1:
                        al_v = al_v[:, feature_idx] if plot_type.lower() in ["sorted_uncertainty", "su"] else np.mean(al_v, axis=-1)
                    al_std = np.sqrt(np.maximum(0, al_v)) / scale
                    mean_al = np.mean(al_std)
                    mean_comb = np.mean(np.sqrt((al_std ** 2) + (ep_std ** 2)))
                    stats_str = (
                        rf"Mean Aleatoric $\sigma$: {mean_al:.4f}" "\n"
                        rf"Mean Epistemic $\sigma$: {mean_ep:.4f}" "\n"
                        rf"Mean Total $\sigma$:     {mean_comb:.4f}"
                    )
                else:
                    stats_str = rf"Mean Epistemic $\sigma$: {mean_ep:.4f}"

                sub_ax.text(
                    0.03, 0.95,
                    stats_str,
                    transform=sub_ax.transAxes,
                    fontsize=12,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.85),
                )

        plt.tight_layout()
        return fig, axes

    def plot_scatter_errorbars(
        self,
        ax,
        model,
        train_yhat,
        test_yhat,
        ytrain=None,
        ytest=None,
        y_idx: int = 0,
        relative: bool = False,
    ):
        """
        Draw scatter plot error bars for UQ models in diagonal_validation_plot
        and validation_plot.

        Parameters
        ----------
        ax: matplotlib.pyplot.axis
            Target axis to draw error bars onto.
        model: UQ model
            The model instance being evaluated.
        train_yhat: np.ndarray
            Predicted outcomes on the training set.
        test_yhat: np.ndarray
            Predicted outcomes on the testing set.
        ytrain: np.ndarray or DataArray
            Actual outcomes on the training set.
        ytest: np.ndarray or DataArray
            Actual outcomes on the testing set.
        y_idx: int
            Target output feature index.
        relative: bool, default=False
            If True, plot relative error percentage error bars (validation_plot).
            If False, plot actual outcome error bars (diagonal_validation_plot).
        """
        if model is not None and model is not self.model:
            unc_train = model.predict_with_uncertainty(self.xtrain.values)
            unc_test = model.predict_with_uncertainty(self.xtest.values)
        else:
            unc_train = self.train_uq_predictions
            unc_test = self.test_uq_predictions

        if unc_train is None or unc_test is None:
            return

        train_ystd = np.sqrt(np.maximum(0, unc_train["epistemic_var"]))
        test_ystd = np.sqrt(np.maximum(0, unc_test["epistemic_var"]))

        if self.yscaler is not None:
            train_ystd = train_ystd / self.yscaler.scale_
            test_ystd = test_ystd / self.yscaler.scale_

        if relative:
            ytest_slice = np.abs(ytest[..., y_idx])
            rel_yerr = (test_ystd[..., y_idx] / ytest_slice) * 100
            errorevery = max(1, ytest.shape[0] // 30)

            cur_ylim = ax.get_ylim()
            ax.errorbar(
                np.arange(ytest.shape[0]),
                np.zeros(ytest.shape[0]),
                yerr=np.ravel(rel_yerr),
                fmt="none",
                ecolor="k",
                alpha=0.4,
                capsize=0,
                elinewidth=1,
                errorevery=errorevery,
            )
            ax.set_ylim(cur_ylim)
        else:
            ax.errorbar(
                np.ravel(train_yhat[..., y_idx]),
                np.ravel(ytrain[..., y_idx]),
                yerr=np.ravel(train_ystd[..., y_idx]),
                fmt="none",
                ecolor="b",
                alpha=0.5,
                capsize=0,
                elinewidth=1,
            )
            ax.errorbar(
                np.ravel(test_yhat[..., y_idx]),
                np.ravel(ytest[..., y_idx]),
                yerr=np.ravel(test_ystd[..., y_idx]),
                fmt="none",
                ecolor="r",
                alpha=0.5,
                capsize=0,
                elinewidth=1,
            )
