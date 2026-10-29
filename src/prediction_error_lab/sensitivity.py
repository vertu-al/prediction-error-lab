"""Robustness analysis for update-rate assumptions in the reversal benchmark."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import SensitivityConfig
from .environments import PiecewiseBernoulliEnvironment
from .learners import FixedRateDeltaLearner, HazardAwareBOCPDLearner, SequentialLearner
from .metrics import compute_recovery_times, compute_seed_metrics, summarise_metrics


@dataclass(frozen=True)
class SensitivityResults:
    """Trial, seed, and summary results for a fixed explicit parameter grid."""

    trials: pd.DataFrame
    seed_metrics: pd.DataFrame
    recovery_times: pd.DataFrame
    summary_metrics: pd.DataFrame


@dataclass(frozen=True)
class SensitivityCondition:
    """One learner family and one value of its adaptive-update parameter."""

    learner: str
    model_family: str
    parameter_name: str
    parameter_value: float


def run_sensitivity(config: SensitivityConfig) -> SensitivityResults:
    """Evaluate each grid condition on the same deterministic outcome streams.

    The two one-dimensional sweeps are intentionally reported separately.  They
    reveal how each learner responds to its own update-rate assumption; they do
    not claim globally fair hyperparameter optimisation between model families.
    """

    base = config.base_config
    environment = PiecewiseBernoulliEnvironment(base.segments)
    conditions = _build_conditions(config)
    records: list[dict[str, float | int | str | bool]] = []

    for seed in base.seeds:
        sequence = environment.sample(seed)
        for condition in conditions:
            learner = _build_learner(base, condition)
            for trial, (outcome, true_probability, change_point) in enumerate(
                zip(
                    sequence.outcomes,
                    sequence.true_probabilities,
                    sequence.change_points,
                    strict=True,
                )
            ):
                prediction = learner.predict()
                learner.update(int(outcome))
                records.append(
                    {
                        "seed": seed,
                        "learner": condition.learner,
                        "model_family": condition.model_family,
                        "parameter_name": condition.parameter_name,
                        "parameter_value": condition.parameter_value,
                        "trial": trial,
                        "outcome": int(outcome),
                        "true_probability": float(true_probability),
                        "change_point": bool(change_point),
                        "prediction": prediction,
                        "prediction_error": float(outcome - prediction),
                    }
                )

    trials = pd.DataFrame.from_records(records)
    metadata = trials.loc[
        :, ["learner", "model_family", "parameter_name", "parameter_value"]
    ].drop_duplicates("learner")
    seed_metrics = _attach_metadata(compute_seed_metrics(trials), metadata)
    recovery_times = _attach_metadata(
        compute_recovery_times(
            trials,
            tolerance=base.recovery_tolerance,
            sustained_trials=base.sustained_trials,
        ),
        metadata,
    )
    summary_metrics = _attach_metadata(summarise_metrics(seed_metrics, recovery_times), metadata)

    return SensitivityResults(
        trials=trials,
        seed_metrics=seed_metrics,
        recovery_times=recovery_times,
        summary_metrics=summary_metrics,
    )


def _build_conditions(config: SensitivityConfig) -> tuple[SensitivityCondition, ...]:
    delta_conditions = tuple(
        SensitivityCondition(
            learner=f"fixed_delta__alpha={value:.6g}",
            model_family="fixed_delta",
            parameter_name="learning_rate",
            parameter_value=value,
        )
        for value in config.delta_learning_rates
    )
    bocpd_conditions = tuple(
        SensitivityCondition(
            learner=f"hazard_aware_bocpd__hazard={value:.6g}",
            model_family="hazard_aware_bocpd",
            parameter_name="hazard",
            parameter_value=value,
        )
        for value in config.bocpd_hazards
    )
    return delta_conditions + bocpd_conditions


def _build_learner(base_config, condition: SensitivityCondition) -> SequentialLearner:
    if condition.model_family == "fixed_delta":
        return FixedRateDeltaLearner(
            learning_rate=condition.parameter_value,
            initial_prediction=base_config.delta_initial_prediction,
        )
    if condition.model_family == "hazard_aware_bocpd":
        return HazardAwareBOCPDLearner(
            hazard=condition.parameter_value,
            alpha0=base_config.bocpd_alpha0,
            beta0=base_config.bocpd_beta0,
            max_run_length=base_config.bocpd_max_run_length,
        )
    raise ValueError(f"Unsupported model family: {condition.model_family}")


def _attach_metadata(table: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    return table.merge(metadata, on="learner", how="left", validate="many_to_one")
