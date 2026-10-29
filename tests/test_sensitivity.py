import pandas as pd

from prediction_error_lab.config import BenchmarkConfig, BernoulliSegment, SensitivityConfig
from prediction_error_lab.sensitivity import run_sensitivity


def _base_config() -> BenchmarkConfig:
    return BenchmarkConfig(
        name="sensitivity-test-base",
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


def test_sensitivity_evaluates_each_condition_on_every_seed() -> None:
    config = SensitivityConfig(
        name="test-sensitivity",
        base_config=_base_config(),
        delta_learning_rates=(0.05, 0.1),
        bocpd_hazards=(0.02, 0.05),
    )

    results = run_sensitivity(config)

    assert results.trials.shape[0] == 2 * 4 * 12
    assert results.seed_metrics.shape[0] == 2 * 4
    assert set(results.seed_metrics["model_family"]) == {"fixed_delta", "hazard_aware_bocpd"}
    assert set(results.seed_metrics["parameter_name"]) == {"learning_rate", "hazard"}
    assert results.summary_metrics.shape[0] == 4 * 4

    repeated = run_sensitivity(config)
    pd.testing.assert_frame_equal(results.seed_metrics, repeated.seed_metrics)
