import numpy as np


def test_observation_excludes_destination_one_hot(env):
    observation = env.task.get_obs()

    expected_size = (
        3 * env.config.num_objects
        + 3 * env.config.num_destinations
        + env.config.num_objects
    )

    assert observation.shape == (expected_size,)


def test_observation_retains_selected_object_identity(env):
    observation = env.task.get_obs()

    object_identity = observation[
        3 * env.config.num_objects
        + 3 * env.config.num_destinations:
    ]

    assert object_identity.shape == (env.config.num_objects,)
    assert np.sum(object_identity) == 1.0
    assert object_identity[env.task.goal_object_idx] == 1.0