"""Evaluation metrics for online binary prediction experiments."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def compute_seed_metrics(trials: pd.DataFrame) -> pd.DataFrame:
    """Calculate outcome and latent-tracking metrics for each seed and learner."""

    required = {"seed", "learner", "prediction", "outcome", "true_probability"}
    _require_columns(trials, required)
    rows: list[dict[str, float | int | str]] = []

    for (seed, learner), group in trials.groupby(["seed", "learner"], sort=True):
        prediction = group["prediction"].to_numpy(dtype=float)
        outcome = group["outcome"].to_numpy(dtype=float)
        target = group["true_probability"].to_numpy(dtype=float)
        clipped = np.clip(prediction, 1e-12, 1.0 - 1e-12)
        rows.append(
            {
                "seed": int(seed),
                "learner": str(learner),
                "brier_score": float(np.mean((prediction - outcome) ** 2)),
                "negative_log_likelihood": float(
                    -np.mean(outcome * np.log(clipped) + (1.0 - outcome) * np.log(1.0 - clipped))
                ),
                "latent_tracking_mae": float(np.mean(np.abs(prediction - target))),
            }
        )

    return pd.DataFrame(rows)


def compute_recovery_times(
    trials: pd.DataFrame,
    tolerance: float,
    sustained_trials: int,
) -> pd.DataFrame:
    """Measure delay until prediction stays close to each new latent probability.

    Recovery is intentionally based on the simulated latent probability.  It is
    therefore a diagnostic benchmark metric, not a quantity one could compute
    directly from an unlabelled real-world stream.
    """

    required = {"seed", "learner", "trial", "prediction", "true_probability", "change_point"}
    _require_columns(trials, required)
    rows: list[dict[str, float | int | str | bool]] = []

    for (seed, learner), group in trials.groupby(["seed", "learner"], sort=True):
        ordered = group.sort_values("trial").reset_index(drop=True)
        change_indices = np.flatnonzero(ordered["change_point"].to_numpy(dtype=bool))
        for change_index in change_indices:
            target = float(ordered.loc[change_index, "true_probability"])
            errors = np.abs(
                ordered.loc[change_index:, "prediction"].to_numpy(dtype=float) - target
            )
            eligible = errors <= tolerance
            recovery_offset = _first_sustained_true_index(eligible, sustained_trials)
            recovered = recovery_offset is not None
            rows.append(
                {
                    "seed": int(seed),
                    "learner": str(learner),
                    "change_trial": int(ordered.loc[change_index, "trial"]),
                    "target_probability": target,
                    "recovered": recovered,
                    "recovery_time": float(recovery_offset) if recovered else math.nan,
                }
            )

    return pd.DataFrame(rows)


def summarise_metrics(seed_metrics: pd.DataFrame, recovery_times: pd.DataFrame) -> pd.DataFrame:
    """Aggregate seed-level metrics with normal-approximation 95% intervals."""

    metric_rows: list[dict[str, float | int | str]] = []
    metric_columns = ["brier_score", "negative_log_likelihood", "latent_tracking_mae"]

    for learner, group in seed_metrics.groupby("learner", sort=True):
        for metric in metric_columns:
            values = group[metric].to_numpy(dtype=float)
            metric_rows.append(_summary_row(str(learner), metric, values))

        learner_recovery = recovery_times.loc[recovery_times["learner"] == learner]
        recovered_values = learner_recovery.loc[
            learner_recovery["recovered"], "recovery_time"
        ].to_numpy(dtype=float)
        recovery_row = _summary_row(str(learner), "recovery_time", recovered_values)
        recovery_row["recovery_rate"] = float(learner_recovery["recovered"].mean())
        metric_rows.append(recovery_row)

    return pd.DataFrame(metric_rows)


def _first_sustained_true_index(values: np.ndarray, length: int) -> int | None:
    if values.size < length:
        return None
    consecutive = 0
    for index, value in enumerate(values):
        consecutive = consecutive + 1 if value else 0
        if consecutive >= length:
            return index - length + 1
    return None


def _summary_row(learner: str, metric: str, values: np.ndarray) -> dict[str, float | int | str]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            "learner": learner,
            "metric": metric,
            "mean": math.nan,
            "std": math.nan,
            "n": 0,
            "ci95_half_width": math.nan,
        }
    std = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
    return {
        "learner": learner,
        "metric": metric,
        "mean": float(np.mean(finite)),
        "std": std,
        "n": int(finite.size),
        "ci95_half_width": float(1.96 * std / np.sqrt(finite.size)),
    }


def _require_columns(frame: pd.DataFrame, required: set[str]) -> None:
    missing = required.difference(frame.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Missing required trial columns: {missing_text}")
