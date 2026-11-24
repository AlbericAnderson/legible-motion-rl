import numpy as np
import pytest

from legible_motion.envs.multi_object import (
    MultiObjectConfig,
    MultiObjectPickAndPlaceEnv,
)


@pytest.fixture
def dense_env():
    env = MultiObjectPickAndPlaceEnv(
        config=MultiObjectConfig(reward_type="dense")
    )
    yield env
    env.close()


@pytest.fixture
def sparse_env():
    env = MultiObjectPickAndPlaceEnv(
        config=MultiObjectConfig(
            reward_type="sparse",
            distance_threshold=0.05,
        )
    )
    yield env
    env.close()


# ------------------------------------------------------------------
# Dense reward
# ------------------------------------------------------------------

def test_dense_reward_equals_negative_euclidean_distance(dense_env):
    achieved_goal = np.array([0.2, 0.0, 0.0])
    goal = np.array([0.0, 0.0, 0.0])

    reward = dense_env.task.compute_reward(
        achieved_goal, goal, {}
    )

    assert reward == pytest.approx(-0.2)


def test_dense_reward_is_zero_at_goal(dense_env):
    goal = np.array([0.1, 0.2, 0.3])

    reward = dense_env.task.compute_reward(
        goal, goal, {}
    )

    assert reward == pytest.approx(0.0)


def test_dense_reward_improves_as_distance_decreases(dense_env):
    goal = np.zeros(3)

    far_reward = dense_env.task.compute_reward(
        np.array([0.2, 0.0, 0.0]),
        goal,
        {},
    )

    close_reward = dense_env.task.compute_reward(
        np.array([0.05, 0.0, 0.0]),
        goal,
        {},
    )

    assert close_reward > far_reward


# ------------------------------------------------------------------
# Sparse reward
# ------------------------------------------------------------------

def test_sparse_reward_is_zero_inside_threshold(sparse_env):
    achieved_goal = np.array([0.02, 0.0, 0.0])
    goal = np.zeros(3)

    reward = sparse_env.task.compute_reward(
        achieved_goal, goal, {}
    )

    assert reward == pytest.approx(0.0)


def test_sparse_reward_is_negative_outside_threshold(sparse_env):
    achieved_goal = np.array([0.1, 0.0, 0.0])
    goal = np.zeros(3)

    reward = sparse_env.task.compute_reward(
        achieved_goal, goal, {}
    )

    assert reward == pytest.approx(-1.0)


def test_sparse_reward_at_threshold_boundary(sparse_env):
    achieved_goal = np.array([0.05, 0.0, 0.0])
    goal = np.zeros(3)

    reward = sparse_env.task.compute_reward(
        achieved_goal, goal, {}
    )

    # PandaGym uses a strict distance < threshold comparison.
    assert reward == pytest.approx(-1.0)


# ------------------------------------------------------------------
# Batched inputs
# ------------------------------------------------------------------

def test_dense_reward_supports_batched_inputs(dense_env):
    achieved_goals = np.array([
        [0.2, 0.0, 0.0],
        [0.1, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ])

    goals = np.zeros((3, 3))

    rewards = dense_env.task.compute_reward(
        achieved_goals, goals, {}
    )

    assert rewards.shape == (3,)
    np.testing.assert_allclose(
        rewards,
        [-0.2, -0.1, 0.0],
    )


def test_sparse_reward_supports_batched_inputs(sparse_env):
    achieved_goals = np.array([
        [0.02, 0.0, 0.0],
        [0.1, 0.0, 0.0],
        [0.05, 0.0, 0.0],
    ])

    goals = np.zeros((3, 3))

    rewards = sparse_env.task.compute_reward(
        achieved_goals, goals, {}
    )

    assert rewards.shape == (3,)
    np.testing.assert_array_equal(
        rewards,
        [0.0, -1.0, -1.0],
    )


# ------------------------------------------------------------------
# Configuration validation
# ------------------------------------------------------------------

def test_invalid_reward_type():
    with pytest.raises(ValueError):
        MultiObjectPickAndPlaceEnv(
            config=MultiObjectConfig(reward_type="invalid")
        )


def test_invalid_distance_threshold():
    with pytest.raises(ValueError):
        MultiObjectPickAndPlaceEnv(
            config=MultiObjectConfig(
                distance_threshold=0.0
            )
        )