"""Synthetic environments used to test sequential prediction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .config import BernoulliSegment


@dataclass(frozen=True)
class BernoulliSequence:
    """A sampled sequence and its known latent probability schedule."""

    outcomes: NDArray[np.int_]
    true_probabilities: NDArray[np.float64]
    change_points: NDArray[np.bool_]


class PiecewiseBernoulliEnvironment:
    """Bernoulli observations with abrupt, known-in-simulation reversals."""

    def __init__(self, segments: tuple[BernoulliSegment, ...]) -> None:
        self.segments = segments

    def sample(self, seed: int) -> BernoulliSequence:
        """Generate one sequence using a local, reproducible random generator."""

        rng = np.random.default_rng(seed)
        n_trials = self.segments[-1].end
        outcomes = np.empty(n_trials, dtype=int)
        probabilities = np.empty(n_trials, dtype=float)
        change_points = np.zeros(n_trials, dtype=bool)

        for index, segment in enumerate(self.segments):
            length = segment.end - segment.start
            probabilities[segment.start : segment.end] = segment.probability
            outcomes[segment.start : segment.end] = rng.binomial(
                1, segment.probability, size=length
            )
            if index > 0:
                change_points[segment.start] = True

        return BernoulliSequence(
            outcomes=outcomes,
            true_probabilities=probabilities,
            change_points=change_points,
        )
