"""Publication-style visual summaries for the benchmark."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LEARNER_LABELS = {
    "fixed_delta": "Fixed-rate delta rule",
    "hazard_aware_bocpd": "Hazard-aware BOCPD",
}
COLORS = {
    "fixed_delta": "#D05A48",
    "hazard_aware_bocpd": "#246A73",
}


def plot_benchmark_summary(
    representative_trace: pd.DataFrame,
    summary_metrics: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Create a trace-plus-metrics figure from already computed results."""

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    figure = plt.figure(figsize=(14, 9), layout="constrained")
    grid = figure.add_gridspec(2, 2, height_ratios=(1.45, 1.0))
    trace_axis = figure.add_subplot(grid[0, :])
    brier_axis = figure.add_subplot(grid[1, 0])
    recovery_axis = figure.add_subplot(grid[1, 1])

    trace = representative_trace.sort_values("trial")
    base = trace.loc[trace["learner"] == "fixed_delta"]
    trace_axis.step(
        base["trial"],
        base["true_probability"],
        where="post",
        color="#202124",
        linewidth=2.2,
        label="Latent reward probability",
    )
    for learner, learner_trace in trace.groupby("learner", sort=True):
        trace_axis.plot(
            learner_trace["trial"],
            learner_trace["prediction"],
            color=COLORS[learner],
            linewidth=1.6,
            alpha=0.92,
            label=LEARNER_LABELS[learner],
        )
    for change_trial in base.loc[base["change_point"], "trial"]:
        trace_axis.axvline(change_trial, color="#7A7A7A", linestyle="--", linewidth=1.0, alpha=0.65)
    trace_axis.set(
        title="Representative sequential predictions after contingency reversals",
        xlabel="Trial",
        ylabel="P(outcome = 1)",
        ylim=(0.0, 1.0),
    )
    trace_axis.legend(frameon=False, ncols=3, loc="upper center")
    trace_axis.spines[["top", "right"]].set_visible(False)

    _plot_summary_bars(
        brier_axis,
        summary_metrics,
        metric="brier_score",
        title="Mean Brier score across seeds",
        ylabel="Lower is better",
    )
    _plot_summary_bars(
        recovery_axis,
        summary_metrics,
        metric="recovery_time",
        title="Mean post-change recovery time",
        ylabel="Trials (lower is better)",
    )
    figure.savefig(destination, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _plot_summary_bars(
    axis: plt.Axes,
    summary: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
) -> None:
    subset = summary.loc[summary["metric"] == metric].sort_values("learner")
    labels = [LEARNER_LABELS[name] for name in subset["learner"]]
    colors = [COLORS[name] for name in subset["learner"]]
    values = subset["mean"].to_numpy(dtype=float)
    errors = subset["ci95_half_width"].to_numpy(dtype=float)
    positions = np.arange(len(subset))
    axis.bar(positions, values, yerr=errors, capsize=5, color=colors, width=0.62)
    axis.set_xticks(positions, labels, rotation=12, ha="right")
    axis.set(title=title, ylabel=ylabel)
    axis.spines[["top", "right"]].set_visible(False)


def plot_sensitivity_summary(
    summary_metrics: pd.DataFrame,
    output_path: str | Path,
    baseline_delta_learning_rate: float,
    baseline_bocpd_hazard: float,
) -> None:
    """Plot how the two model families respond to explicit update-rate sweeps."""

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(13, 8.5), layout="constrained")
    panel_specs = (
        (
            "fixed_delta",
            "brier_score",
            "Brier score",
            "Learning rate α",
            baseline_delta_learning_rate,
        ),
        (
            "hazard_aware_bocpd",
            "brier_score",
            "Brier score",
            "Change-point hazard H",
            baseline_bocpd_hazard,
        ),
        (
            "fixed_delta",
            "recovery_time",
            "Recovery time (trials)",
            "Learning rate α",
            baseline_delta_learning_rate,
        ),
        (
            "hazard_aware_bocpd",
            "recovery_time",
            "Recovery time (trials)",
            "Change-point hazard H",
            baseline_bocpd_hazard,
        ),
    )

    for axis, (family, metric, ylabel, xlabel, baseline) in zip(
        axes.flat, panel_specs, strict=True
    ):
        subset = summary_metrics.loc[
            (summary_metrics["model_family"] == family)
            & (summary_metrics["metric"] == metric)
        ].sort_values("parameter_value")
        axis.errorbar(
            subset["parameter_value"],
            subset["mean"],
            yerr=subset["ci95_half_width"],
            color=COLORS[family],
            marker="o",
            linewidth=2.0,
            capsize=4,
            label="mean ± 95% CI",
        )
        parameter_symbol = "α" if family == "fixed_delta" else "H"
        axis.axvline(
            baseline,
            color="#6B6B6B",
            linestyle="--",
            linewidth=1.1,
            label=f"baseline {parameter_symbol} = {baseline:.3g}",
        )
        family_label = LEARNER_LABELS[family]
        axis.set(title=family_label, xlabel=xlabel, ylabel=ylabel)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, loc="best")

    figure.suptitle(
        "Sensitivity to update-rate assumptions (identical seeded outcome streams)",
        fontsize=14,
        fontweight="bold",
    )
    figure.savefig(destination, dpi=220, bbox_inches="tight")
    plt.close(figure)
