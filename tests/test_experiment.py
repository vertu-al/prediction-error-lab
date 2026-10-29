import pandas as pd

from prediction_error_lab.config import BenchmarkConfig, BernoulliSegment
from prediction_error_lab.experiment import run_experiment


def _small_config() -> BenchmarkConfig:
    return BenchmarkConfig(
        name="test-benchmark",
        seeds=(2, 3),
        segments=(
            BernoulliSegment(0, 6, 0.8),
            BernoulliSegment(6, 12, 0.2),
        ),
        delta_learning_rate=0.1,
        delta_initial_prediction=0.5,
        bocpd_hazard=0.05,
        bocpd_alpha0=1.0,
        bocpd_beta0=1.0,
        bocpd_max_run_length=10,
        recovery_tolerance=0.2,
        sustained_trials=2,
        representative_seed=2,
    )


def test_experiment_is_reproducible_and_has_expected_shape() -> None:
    first = run_experiment(_small_config())
    second = run_experiment(_small_config())

    assert first.trials.shape[0] == 2 * 2 * 12
    assert set(first.trials["learner"]) == {"fixed_delta", "hazard_aware_bocpd"}
    pd.testing.assert_frame_equal(first.seed_metrics, second.seed_metrics)
    pd.testing.assert_frame_equal(first.summary_metrics, second.summary_metrics)
