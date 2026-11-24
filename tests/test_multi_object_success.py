import numpy as np

from legible_motion.envs import MultiObjectConfig


IDENTITY = np.array([0.0, 0.0, 0.0, 1.0])


def selected_body_ids(env):
    return env.task._get_selected_body_ids()


def place_selected_object_at_destination(env):
    """Place the selected object at the center of its destination."""
    task = env.task

    object_name = task.object_names[task.goal_object_idx]
    destination_name = task.destination_names[
        task.goal_destination_idx
    ]

    position = np.asarray(
        env.sim.get_base_position(destination_name)
    ).copy()

    position[2] = task.config.object_size / 2

    env.sim.set_base_pose(
        object_name,
        position,
        IDENTITY,
    )

    # Allow the object to settle physically.
    for _ in range(20):
        env.sim.step()


def test_object_containment(env):
    task = env.task
    physics = env.sim.physics_client

    object_id, destination_id, _, _ = selected_body_ids(env)

    place_selected_object_at_destination(env)

    assert task._is_object_inside_destination(
        physics, object_id, destination_id
    )

    # Move the object far enough that its footprint leaves the target.
    object_name = task.object_names[task.goal_object_idx]
    position = np.asarray(
        env.sim.get_base_position(object_name)
    ).copy()

    position[0] += 0.03

    env.sim.set_base_pose(object_name, position, IDENTITY)

    assert not task._is_object_inside_destination(
        physics, object_id, destination_id
    )


def test_object_must_be_supported_by_table(env):
    task = env.task
    physics = env.sim.physics_client

    object_id, _, table_id, _ = selected_body_ids(env)

    for _ in range(10):
        env.sim.step()

    assert task._is_object_supported(
        physics, object_id, table_id
    )

    # Lift the object away from the table.
    object_name = task.object_names[task.goal_object_idx]
    position = np.asarray(
        env.sim.get_base_position(object_name)
    ).copy()

    position[2] += 0.10

    env.sim.set_base_pose(object_name, position, IDENTITY)
    env.sim.step()

    assert not task._is_object_supported(
        physics, object_id, table_id
    )


def test_gripper_release_detection():
    """Test finger-contact interpretation independently of PyBullet."""

    class FakePhysics:
        def __init__(self, contacts):
            self.contacts = contacts

        def getContactPoints(self, **kwargs):
            return self.contacts

    object_id = 3
    robot_id = 0

    no_contacts = FakePhysics([])

    assert task_gripper_released(
        no_contacts, object_id, robot_id
    )

    # PyBullet contact tuple: index 4 is the bodyB link index.
    finger_contact = (
        0, object_id, robot_id, -1, 9,
        None, None, None, None, 1.0,
    )

    contact_physics = FakePhysics([finger_contact])

    assert not task_gripper_released(
        contact_physics, object_id, robot_id
    )


def task_gripper_released(physics, object_id, robot_id):
    """Apply the same finger-link rule as the task implementation."""
    contacts = physics.getContactPoints(
        bodyA=object_id,
        bodyB=robot_id,
    )

    finger_links = {9, 10}

    return not any(
        contact[4] in finger_links
        for contact in contacts
    )


def test_object_must_be_stationary(env):
    task = env.task
    physics = env.sim.physics_client

    object_id, _, _, _ = selected_body_ids(env)

    for _ in range(10):
        env.sim.step()

    assert task._is_object_stationary(physics, object_id)

    physics.resetBaseVelocity(
        object_id,
        linearVelocity=[0.1, 0.0, 0.0],
        angularVelocity=[0.0, 0.0, 0.0],
    )

    assert not task._is_object_stationary(physics, object_id)


def test_success_requires_consecutive_valid_steps(env):
    task = env.task

    place_selected_object_at_destination(env)

    for step in range(task.config.success_stability_steps - 1):
        env.sim.step()

        success = task.is_success(
            task.get_achieved_goal(),
            task.get_goal(),
        )

        assert not bool(success), f"Unexpected success at step {step + 1}"

    env.sim.step()

    success = task.is_success(
        task.get_achieved_goal(),
        task.get_goal(),
    )

    assert bool(success)
    assert task._success_streak == task.config.success_stability_steps


def test_invalid_placement_resets_success_streak(env):
    task = env.task

    place_selected_object_at_destination(env)

    for _ in range(4):
        env.sim.step()
        task.is_success(
            task.get_achieved_goal(),
            task.get_goal(),
        )

    assert task._success_streak == 4

    object_name = task.object_names[task.goal_object_idx]
    position = np.asarray(
        env.sim.get_base_position(object_name)
    ).copy()

    position[0] += 0.10

    env.sim.set_base_pose(object_name, position, IDENTITY)
    env.sim.step()

    success = task.is_success(
        task.get_achieved_goal(),
        task.get_goal(),
    )

    assert not bool(success)
    assert task._success_streak == 0


def test_reset_clears_success_streak(env):
    task = env.task

    place_selected_object_at_destination(env)

    for _ in range(4):
        env.sim.step()
        task.is_success(
            task.get_achieved_goal(),
            task.get_goal(),
        )

    assert task._success_streak == 4

    env.reset(seed=123)

    assert task._success_streak == 0


def test_only_selected_object_determines_success(env):
    task = env.task

    selected_idx = task.goal_object_idx
    other_idx = (selected_idx + 1) % task.config.num_objects

    other_object_name = task.object_names[other_idx]
    destination_name = task.destination_names[
        task.goal_destination_idx
    ]

    position = np.asarray(
        env.sim.get_base_position(destination_name)
    ).copy()

    position[2] = task.config.object_size / 2

    env.sim.set_base_pose(
        other_object_name,
        position,
        IDENTITY,
    )

    for _ in range(20):
        env.sim.step()

    success = task.is_success(
        task.get_achieved_goal(),
        task.get_goal(),
    )

    assert not bool(success)
    assert task._success_streak == 0


def test_success_configuration_defaults():
    config = MultiObjectConfig()

    assert config.success_stability_steps == 10
    assert config.placement_margin == 0.001
    assert config.linear_velocity_threshold == 0.02
    assert config.angular_velocity_threshold == 0.1