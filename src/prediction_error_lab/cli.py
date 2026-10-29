"""Command-line entry point for a fully reproducible benchmark run."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

from .config import load_config
from .experiment import ExperimentResults, run_experiment
from .plotting import plot_benchmark_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the prediction-error reversal benchmark.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a JSON benchmark config.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory for tables and figures (default: results).",
    )
    parser.add_argument(
        "--save-trials",
        action="store_true",
        help="Also write all trial-level rows; otherwise only compact derived artefacts are saved.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    results = run_experiment(config)
    _write_results(results, args.output_dir, args.config, save_trials=args.save_trials)


def _write_results(
    results: ExperimentResults,
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
    results.representative_trace.to_csv(output_dir / "representative_trace.csv", index=False)
    if save_trials:
        results.trials.to_csv(output_dir / "trial_level_results.csv", index=False)

    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    pd.DataFrame(
        [
            {
                "config_file": config_path.name,
                "config_sha256": config_hash,
                "n_seed_learner_runs": int(results.seed_metrics.shape[0]),
                "n_trial_rows": int(results.trials.shape[0]),
                "trial_level_results_saved": save_trials,
            }
        ]
    ).to_csv(output_dir / "experiment_log.csv", index=False)
    plot_benchmark_summary(
        results.representative_trace,
        results.summary_metrics,
        figures_dir / "reversal_benchmark_summary.png",
    )


if __name__ == "__main__":
    main()
