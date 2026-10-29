"""Reproducible benchmarks for sequential prediction-error learners."""

from .config import BenchmarkConfig, load_config
from .experiment import ExperimentResults, run_experiment

__all__ = ["BenchmarkConfig", "ExperimentResults", "load_config", "run_experiment"]
