import numpy as np

from prediction_error_lab.config import BernoulliSegment
from prediction_error_lab.environments import PiecewiseBernoulliEnvironment


def test_environment_marks_only_real_segment_boundaries() -> None:
    environment = PiecewiseBernoulliEnvironment(
        (
            BernoulliSegment(0, 3, 0.8),
            BernoulliSegment(3, 5, 0.2),
        )
    )

    sequence = environment.sample(seed=9)

    assert sequence.outcomes.shape == (5,)
    assert np.allclose(sequence.true_probabilities, [0.8, 0.8, 0.8, 0.2, 0.2])
    assert np.flatnonzero(sequence.change_points).tolist() == [3]


def test_environment_is_reproducible_for_a_seed() -> None:
    environment = PiecewiseBernoulliEnvironment((BernoulliSegment(0, 12, 0.6),))

    first = environment.sample(seed=17)
    second = environment.sample(seed=17)

    assert np.array_equal(first.outcomes, second.outcomes)
