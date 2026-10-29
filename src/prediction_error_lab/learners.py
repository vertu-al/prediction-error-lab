"""Sequential learners for binary outcomes."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class SequentialLearner(ABC):
    """Minimal interface shared by all online learners in this benchmark."""

    name: str

    @abstractmethod
    def reset(self) -> None:
        """Return the learner to its initial state."""

    @abstractmethod
    def predict(self) -> float:
        """Return P(outcome = 1) before seeing the next observation."""

    @abstractmethod
    def update(self, observation: int) -> None:
        """Assimilate a binary observation after prediction."""


class FixedRateDeltaLearner(SequentialLearner):
    """Rescorla-Wagner style learner with a constant learning rate."""

    name = "fixed_delta"

    def __init__(self, learning_rate: float, initial_prediction: float = 0.5) -> None:
        if not 0.0 < learning_rate <= 1.0:
            raise ValueError("learning_rate must be in (0, 1].")
        if not 0.0 < initial_prediction < 1.0:
            raise ValueError("initial_prediction must be in (0, 1).")
        self.learning_rate = learning_rate
        self.initial_prediction = initial_prediction
        self.reset()

    def reset(self) -> None:
        self.value = self.initial_prediction

    def predict(self) -> float:
        return float(self.value)

    def update(self, observation: int) -> None:
        _validate_binary_observation(observation)
        prediction_error = observation - self.value
        self.value += self.learning_rate * prediction_error


class HazardAwareBOCPDLearner(SequentialLearner):
    """Beta-Bernoulli Bayesian online change-point detector.

    The learner tracks a posterior over run lengths.  A constant hazard gives
    each new observation a prior chance of beginning a new latent segment.
    This is a deliberately compact reference implementation inspired by Adams
    & MacKay (2007), rather than a production BOCPD library or a claim of
    biological process realism.
    """

    name = "hazard_aware_bocpd"

    def __init__(
        self,
        hazard: float,
        alpha0: float = 1.0,
        beta0: float = 1.0,
        max_run_length: int = 250,
    ) -> None:
        if not 0.0 < hazard < 1.0:
            raise ValueError("hazard must be in (0, 1).")
        if alpha0 <= 0.0 or beta0 <= 0.0:
            raise ValueError("Beta prior parameters must be positive.")
        if max_run_length < 1:
            raise ValueError("max_run_length must be at least 1.")
        self.hazard = hazard
        self.alpha0 = alpha0
        self.beta0 = beta0
        self.max_run_length = max_run_length
        self.reset()

    def reset(self) -> None:
        self.run_length_probabilities: NDArray[np.float64] = np.array([1.0], dtype=float)
        self.alphas: NDArray[np.float64] = np.array([self.alpha0], dtype=float)
        self.betas: NDArray[np.float64] = np.array([self.beta0], dtype=float)
        self.last_change_probability = 0.0

    def predict(self) -> float:
        means = self.alphas / (self.alphas + self.betas)
        return float(np.dot(self.run_length_probabilities, means))

    def update(self, observation: int) -> None:
        _validate_binary_observation(observation)
        predictive_means = self.alphas / (self.alphas + self.betas)
        likelihoods = np.where(observation == 1, predictive_means, 1.0 - predictive_means)
        joint = self.run_length_probabilities * likelihoods

        next_probabilities = np.empty(joint.size + 1, dtype=float)
        next_probabilities[0] = self.hazard * joint.sum()
        next_probabilities[1:] = (1.0 - self.hazard) * joint

        next_alphas = np.empty_like(next_probabilities)
        next_betas = np.empty_like(next_probabilities)
        next_alphas[0] = self.alpha0 + observation
        next_betas[0] = self.beta0 + (1 - observation)
        next_alphas[1:] = self.alphas + observation
        next_betas[1:] = self.betas + (1 - observation)

        normalizer = next_probabilities.sum()
        if normalizer <= 0.0 or not np.isfinite(normalizer):
            raise FloatingPointError("BOCPD posterior could not be normalized.")
        next_probabilities /= normalizer

        max_states = self.max_run_length + 1
        if next_probabilities.size > max_states:
            next_probabilities = next_probabilities[:max_states]
            next_alphas = next_alphas[:max_states]
            next_betas = next_betas[:max_states]
            next_probabilities /= next_probabilities.sum()

        self.last_change_probability = float(next_probabilities[0])
        self.run_length_probabilities = next_probabilities
        self.alphas = next_alphas
        self.betas = next_betas


def _validate_binary_observation(observation: int) -> None:
    if observation not in (0, 1):
        raise ValueError("This benchmark only accepts binary observations 0 or 1.")
