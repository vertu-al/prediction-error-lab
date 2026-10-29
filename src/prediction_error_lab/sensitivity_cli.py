"""Command-line entry point for the explicit update-rate sensitivity analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from .config import SensitivityConfig, load_sensitivity_config
from .plotting import plot_sensitivity_summary
from .sensitivity import SensitivityResults, run_sensitivity


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the prediction-error update-rate sensitivity analysis."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a sensitivity JSON config.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/sensitivity"),
        help="Directory for sensitivity tables and figure (default: results/sensitivity).",
    )
    parser.add_argument(
        "--save-trials",
        action="store_true",
        help="Also write all trial-level rows; otherwise only compact derived artefacts are saved.",
    )
    args = parser.parse_args()

    config = load_sensitivity_config(args.config)
    results = run_sensitivity(config)
    _write_results(results, config, args.output_dir, args.config, save_trials=args.save_trials)


def _write_results(
    results: SensitivityResults,
    config: SensitivityConfig,
    output_dir: Path,
    config_path: Path,
    save_trials: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    results.seed_metrics.to_csv(output_dir / "seed_metrics.csv", index=False)
    results.recovery_times.to_csv(output_dir / "recovery_times.csv", index=False)
    results.summary_metrics.to_csv(output_dir / "summary_metrics.csv", index=False)
    if save_trials:
        results.trials.to_csv(output_dir / "trial_level_results.csv", index=False)

    raw_sensitivity = json.loads(config_path.read_text(encoding="utf-8"))
    base_config_path = config_path.parent / str(raw_sensitivity["base_config"])
    pd.DataFrame(
        [
            {
                "sensitivity_config_file": config_path.name,
                "sensitivity_config_sha256": _sha256(config_path),
                "base_config_file": base_config_path.name,
                "base_config_sha256": _sha256(base_config_path),
                "n_conditions": int(
                    len(config.delta_learning_rates) + len(config.bocpd_hazards)
                ),
                "n_seed_condition_runs": int(results.seed_metrics.shape[0]),
                "n_trial_rows": int(results.trials.shape[0]),
                "trial_level_results_saved": save_trials,
            }
        ]
    ).to_csv(output_dir / "experiment_log.csv", index=False)
    plot_sensitivity_summary(
        results.summary_metrics,
        figures_dir / "update_rate_sensitivity.png",
        baseline_delta_learning_rate=config.base_config.delta_learning_rate,
        baseline_bocpd_hazard=config.base_config.bocpd_hazard,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
