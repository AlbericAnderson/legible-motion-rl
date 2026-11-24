import pytest

from legible_motion.envs.multi_object import (
    MultiObjectConfig,
    MultiObjectPickAndPlaceEnv,
)


def test_episode_counter_resets_on_reset():
    env = MultiObjectPickAndPlaceEnv(
        config=MultiObjectConfig(max_episode_steps=5)
    )

    try:
        env.reset(seed=42)
        env.step(env.action_space.sample())

        assert env.task._elapsed_steps == 1

        env.reset(seed=43)

        assert env.task._elapsed_steps == 0
    finally:
        env.close()


def test_episode_truncates_at_horizon():
    env = MultiObjectPickAndPlaceEnv(
        config=MultiObjectConfig(max_episode_steps=3)
    )

    try:
        env.reset(seed=42)

        for _ in range(2):
            _, _, terminated, truncated, _ = env.step(
                env.action_space.sample()
            )
            assert not terminated
            assert not truncated

        _, _, terminated, truncated, _ = env.step(
            env.action_space.sample()
        )

        assert not terminated
        assert truncated
        assert env.task._elapsed_steps == 3
    finally:
        env.close()


def test_success_takes_precedence_over_timeout():
    env = MultiObjectPickAndPlaceEnv(
        config=MultiObjectConfig(max_episode_steps=1)
    )

    try:
        env.reset(seed=42)

        # Simulate successful completion on the final allowed step.
        env.task.is_success = lambda *args: True

        _, _, terminated, truncated, info = env.step(
            env.action_space.sample()
        )

        assert terminated
        assert not truncated
        assert info["is_success"]
    finally:
        env.close()


def test_invalid_episode_horizon():
    with pytest.raises(ValueError):
        MultiObjectPickAndPlaceEnv(
            config=MultiObjectConfig(max_episode_steps=0)
        )