"""Configuration loading and validation for the reversal benchmark."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BernoulliSegment:
    """A contiguous interval with a fixed Bernoulli reward probability."""

    start: int
    end: int
    probability: float

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError("Segment bounds must satisfy 0 <= start < end.")
        if not 0.0 < self.probability < 1.0:
            raise ValueError("Bernoulli probabilities must lie strictly between 0 and 1.")


@dataclass(frozen=True)
class BenchmarkConfig:
    """All parameters required to run one benchmark deterministically."""

    name: str
    seeds: tuple[int, ...]
    segments: tuple[BernoulliSegment, ...]
    delta_learning_rate: float
    delta_initial_prediction: float
    bocpd_hazard: float
    bocpd_alpha0: float
    bocpd_beta0: float
    bocpd_max_run_length: int
    recovery_tolerance: float
    sustained_trials: int
    representative_seed: int

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Benchmark name must not be empty.")
        if not self.seeds:
            raise ValueError("At least one random seed is required.")
        if not 0.0 < self.delta_learning_rate <= 1.0:
            raise ValueError("delta_learning_rate must be in (0, 1].")
        if not 0.0 < self.delta_initial_prediction < 1.0:
            raise ValueError("delta_initial_prediction must be in (0, 1).")
        if not 0.0 < self.bocpd_hazard < 1.0:
            raise ValueError("bocpd_hazard must be in (0, 1).")
        if self.bocpd_alpha0 <= 0.0 or self.bocpd_beta0 <= 0.0:
            raise ValueError("Beta prior parameters must be positive.")
        if self.bocpd_max_run_length < 1:
            raise ValueError("bocpd_max_run_length must be at least 1.")
        if not 0.0 < self.recovery_tolerance < 1.0:
            raise ValueError("recovery_tolerance must be in (0, 1).")
        if self.sustained_trials < 1:
            raise ValueError("sustained_trials must be at least 1.")
        if self.representative_seed not in self.seeds:
            raise ValueError("representative_seed must be one of seeds.")
        _validate_segments(self.segments)

    @property
    def n_trials(self) -> int:
        """Total number of observations generated per seed."""

        return self.segments[-1].end


@dataclass(frozen=True)
class SensitivityConfig:
    """Explicit hyperparameter grids for robustness checks around one benchmark."""

    name: str
    base_config: BenchmarkConfig
    delta_learning_rates: tuple[float, ...]
    bocpd_hazards: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Sensitivity analysis name must not be empty.")
        _validate_probability_grid(
            self.delta_learning_rates,
            label="delta_learning_rates",
            upper_inclusive=True,
        )
        _validate_probability_grid(
            self.bocpd_hazards,
            label="bocpd_hazards",
            upper_inclusive=False,
        )


def _validate_segments(segments: tuple[BernoulliSegment, ...]) -> None:
    if not segments:
        raise ValueError("At least one environmental segment is required.")
    if segments[0].start != 0:
        raise ValueError("The first segment must start at trial 0.")
    for previous, current in zip(segments, segments[1:], strict=False):
        if previous.end != current.start:
            raise ValueError("Segments must be contiguous and non-overlapping.")


def load_config(path: str | Path) -> BenchmarkConfig:
    """Load a JSON benchmark configuration into a validated dataclass."""

    config_path = Path(path)
    raw = json.loads(config_path.read_text(encoding="utf-8"))

    segments = tuple(
        BernoulliSegment(
            start=int(segment["start"]),
            end=int(segment["end"]),
            probability=float(segment["probability"]),
        )
        for segment in raw["environment"]["segments"]
    )
    learners = raw["learners"]
    metrics = raw["metrics"]

    return BenchmarkConfig(
        name=str(raw["name"]),
        seeds=tuple(int(seed) for seed in raw["seeds"]),
        segments=segments,
        delta_learning_rate=float(learners["fixed_delta"]["learning_rate"]),
        delta_initial_prediction=float(learners["fixed_delta"]["initial_prediction"]),
        bocpd_hazard=float(learners["hazard_aware_bocpd"]["hazard"]),
        bocpd_alpha0=float(learners["hazard_aware_bocpd"]["alpha0"]),
        bocpd_beta0=float(learners["hazard_aware_bocpd"]["beta0"]),
        bocpd_max_run_length=int(learners["hazard_aware_bocpd"]["max_run_length"]),
        recovery_tolerance=float(metrics["recovery_tolerance"]),
        sustained_trials=int(metrics["sustained_trials"]),
        representative_seed=int(raw["representative_seed"]),
    )


def load_sensitivity_config(path: str | Path) -> SensitivityConfig:
    """Load a grid definition and its referenced base benchmark configuration."""

    config_path = Path(path)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    base_config_path = config_path.parent / str(raw["base_config"])

    return SensitivityConfig(
        name=str(raw["name"]),
        base_config=load_config(base_config_path),
        delta_learning_rates=tuple(float(value) for value in raw["fixed_delta_learning_rates"]),
        bocpd_hazards=tuple(float(value) for value in raw["bocpd_hazards"]),
    )


def _validate_probability_grid(
    values: tuple[float, ...],
    label: str,
    upper_inclusive: bool,
) -> None:
    if not values:
        raise ValueError(f"{label} must include at least one value.")
    if len(set(values)) != len(values):
        raise ValueError(f"{label} must not contain duplicate values.")
    upper_ok = (lambda value: value <= 1.0) if upper_inclusive else (lambda value: value < 1.0)
    if any(value <= 0.0 or not upper_ok(value) for value in values):
        bound = "(0, 1]" if upper_inclusive else "(0, 1)"
        raise ValueError(f"{label} values must lie in {bound}.")
