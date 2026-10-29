import numpy as np

from prediction_error_lab.learners import FixedRateDeltaLearner, HazardAwareBOCPDLearner


def test_delta_rule_applies_prediction_error_update() -> None:
    learner = FixedRateDeltaLearner(learning_rate=0.2, initial_prediction=0.5)

    learner.update(1)

    assert learner.predict() == 0.6


def test_bocpd_keeps_a_normalised_posterior() -> None:
    learner = HazardAwareBOCPDLearner(hazard=0.05, max_run_length=10)

    for outcome in [1, 1, 0, 1, 0, 0, 1]:
        learner.update(outcome)
        assert np.isclose(learner.run_length_probabilities.sum(), 1.0)
        assert learner.run_length_probabilities.size <= 11
        assert 0.0 < learner.predict() < 1.0


def test_bocpd_matches_hand_calculated_two_observation_posterior() -> None:
    learner = HazardAwareBOCPDLearner(hazard=0.05)

    learner.update(1)
    learner.update(0)

    np.testing.assert_allclose(
        learner.run_length_probabilities,
        [0.05, 0.0475, 0.9025],
    )
    assert np.isclose(learner.predict(), 59.0 / 120.0)