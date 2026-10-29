"""Benchmark orchestration without notebook-only state."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import BenchmarkConfig
from .environments import PiecewiseBernoulliEnvironment
from .learners import FixedRateDeltaLearner, HazardAwareBOCPDLearner, SequentialLearner
from .metrics import compute_recovery_times, compute_seed_metrics, summarise_metrics


@dataclass(frozen=True)
class ExperimentResults:
    """In-memory results from a complete multi-seed benchmark."""

    trials: pd.DataFrame
    seed_metrics: pd.DataFrame
    recovery_times: pd.DataFrame
    summary_metrics: pd.DataFrame
    representative_trace: pd.DataFrame


def run_experiment(config: BenchmarkConfig) -> ExperimentResults:
    """Run both learners on identical seeded observation sequences."""

    environment = PiecewiseBernoulliEnvironment(config.segments)
    records: list[dict[str, float | int | str | bool]] = []

    for seed in config.seeds:
        sequence = environment.sample(seed)
        for learner in _build_learners(config):
            learner.reset()
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
                        "learner": learner.name,
                        "trial": trial,
                        "outcome": int(outcome),
                        "true_probability": float(true_probability),
                        "change_point": bool(change_point),
                        "prediction": prediction,
                        "prediction_error": float(outcome - prediction),
                    }
                )

    trials = pd.DataFrame.from_records(records)
    seed_metrics = compute_seed_metrics(trials)
    recovery_times = compute_recovery_times(
        trials,
        tolerance=config.recovery_tolerance,
        sustained_trials=config.sustained_trials,
    )
    summary_metrics = summarise_metrics(seed_metrics, recovery_times)
    representative_trace = trials.loc[trials["seed"] == config.representative_seed].copy()

    return ExperimentResults(
        trials=trials,
        seed_metrics=seed_metrics,
        recovery_times=recovery_times,
        summary_metrics=summary_metrics,
        representative_trace=representative_trace,
    )


def _build_learners(config: BenchmarkConfig) -> tuple[SequentialLearner, ...]:
    return (
        FixedRateDeltaLearner(
            learning_rate=config.delta_learning_rate,
            initial_prediction=config.delta_initial_prediction,
        ),
        HazardAwareBOCPDLearner(
            hazard=config.bocpd_hazard,
            alpha0=config.bocpd_alpha0,
            beta0=config.bocpd_beta0,
            max_run_length=config.bocpd_max_run_length,
        ),
    )
