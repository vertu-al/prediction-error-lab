import math

import pandas as pd
import pytest

from prediction_error_lab.metrics import compute_recovery_times, compute_seed_metrics


def test_recovery_time_requires_sustained_accuracy() -> None:
    trials = pd.DataFrame(
        {
            "seed": [0] * 7,
            "learner": ["demo"] * 7,
            "trial": list(range(7)),
            "prediction": [0.2, 0.2, 0.5, 0.78, 0.70, 0.79, 0.82],
            "true_probability": [0.2, 0.2, 0.8, 0.8, 0.8, 0.8, 0.8],
            "change_point": [False, False, True, False, False, False, False],
        }
    )

    recovery = compute_recovery_times(trials, tolerance=0.05, sustained_trials=2)

    assert len(recovery) == 1
    assert bool(recovery.loc[0, "recovered"])
    assert recovery.loc[0, "recovery_time"] == 3.0


def test_seed_metrics_match_hand_calculation() -> None:
    trials = pd.DataFrame(
        {
            "seed": [0, 0],
            "learner": ["demo", "demo"],
            "prediction": [0.8, 0.2],
            "outcome": [1, 0],
            "true_probability": [0.75, 0.25],
        }
    )

    metrics = compute_seed_metrics(trials)
    row = metrics.iloc[0]

    assert row["brier_score"] == pytest.approx(0.04)
    assert row["negative_log_likelihood"] == pytest.approx(-math.log(0.8))
    assert row["latent_tracking_mae"] == pytest.approx(0.05)