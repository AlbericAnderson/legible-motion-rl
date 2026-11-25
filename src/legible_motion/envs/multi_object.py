from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from panda_gym.envs.core import RobotTaskEnv, Task
from panda_gym.envs.robots.panda import Panda
from panda_gym.pybullet import PyBullet


@dataclass
class MultiObjectConfig:
    num_objects: int = 2
    num_destinations: int = 2

    object_size: float = 0.04
    object_mass: float = 0.1

    workspace_low: tuple = (-0.15, -0.15)
    workspace_high: tuple = (0.15, 0.15)

    min_separation: float = 0.10

    table_length: float = 1.1
    table_width: float = 0.7
    table_height: float = 0.4
    
    # Success detection parameters
    destination_half_extents: tuple = (0.035, 0.035)

    placement_margin: float = 0.001
    support_force_threshold: float = 0.01

    linear_velocity_threshold: float = 0.02
    angular_velocity_threshold: float = 0.1

    success_stability_steps: int = 10
    
    # Termination parameters
    max_episode_steps: int = 250
    
    # Reward parameters
    reward_type: str = "dense"
    distance_threshold: float = 0.05


class MultiObjectPickAndPlaceTask(Task):
    def __init__(
        self,
        sim: PyBullet,
        config: MultiObjectConfig,
    ):
        super().__init__(sim)

        if config.num_objects < 1:
            raise ValueError("At least one object is required.")

        if config.num_destinations < 1:
            raise ValueError("At least one destination is required.")
        
        if config.max_episode_steps < 1:
            raise ValueError("max_episode_steps must be at least 1.")
        
        if config.reward_type not in {"dense", "sparse"}:
            raise ValueError("reward_type must be 'dense' or 'sparse'.")
        
        if config.distance_threshold <= 0:
            raise ValueError("distance_threshold must be positive.")

        self.config = config
        self.object_names = [
            f"object_{i}" for i in range(config.num_objects)
        ]
        self.destination_names = [
            f"target_{i}" for i in range(config.num_destinations)
        ]

        self.object_positions = np.zeros(
            (config.num_objects, 3), dtype=np.float32
        )
        self.destination_positions = np.zeros(
            (config.num_destinations, 3), dtype=np.float32
        )

        self.goal_object_idx = 0
        self.goal_destination_idx = 0
        self._success_streak = 0
        self._elapsed_steps = 0

        with self.sim.no_rendering():
            self._create_scene()

    def _create_scene(self):
        """Create the shared table, objects, and destination markers."""

        self.sim.create_plane(z_offset=-self.config.table_height)

        self.sim.create_table(
            length=self.config.table_length,
            width=self.config.table_width,
            height=self.config.table_height,
            x_offset=-0.3,
        )

        object_colors = [
            np.array([0.1, 0.3, 0.95, 1.0]),
            np.array([0.95, 0.25, 0.1, 1.0]),
            np.array([0.1, 0.8, 0.3, 1.0]),
            np.array([0.9, 0.75, 0.1, 1.0]),
        ]

        destination_colors = [
            np.array([0.1, 0.3, 0.95, 0.3]),
            np.array([0.95, 0.25, 0.1, 0.3]),
            np.array([0.1, 0.8, 0.3, 0.3]),
            np.array([0.9, 0.75, 0.1, 0.3]),
        ]

        for i, name in enumerate(self.object_names):
            color = object_colors[i % len(object_colors)]

            self.sim.create_box(
                body_name=name,
                half_extents=np.ones(3) * self.config.object_size / 2,
                mass=self.config.object_mass,
                position=np.array([0.0, 0.0, self.config.object_size / 2]),
                rgba_color=color,
            )

        for i, name in enumerate(self.destination_names):
            color = destination_colors[i % len(destination_colors)]

            self.sim.create_box(
                body_name=name,
                half_extents=np.array([
                    *self.config.destination_half_extents,
                    0.001,
                ]),
                mass=0.0,
                ghost=True,
                position=np.array([0.0, 0.0, 0.001]),
                rgba_color=color,
            )

    def _sample_positions(self, count: int) -> np.ndarray:
        """Sample separated XY positions within the workspace."""

        low = np.asarray(self.config.workspace_low)
        high = np.asarray(self.config.workspace_high)

        positions = []
        max_attempts = 1000

        for _ in range(count):
            for _ in range(max_attempts):
                candidate = self.np_random.uniform(low, high)

                if all(
                    np.linalg.norm(candidate - existing)
                    >= self.config.min_separation
                    for existing in positions
                ):
                    positions.append(candidate)
                    break
            else:
                raise RuntimeError(
                    "Could not sample sufficiently separated positions. "
                    "Increase workspace size or reduce min_separation."
                )

        return np.asarray(positions, dtype=np.float32)

    def reset(self):
        
        self._success_streak = 0
        self._elapsed_steps = 0
        
        total = (
            self.config.num_objects
            + self.config.num_destinations
        )

        positions = self._sample_positions(total)

        object_xy = positions[:self.config.num_objects]
        destination_xy = positions[self.config.num_objects:]

        z = self.config.object_size / 2

        self.object_positions = np.column_stack(
            [object_xy, np.full(self.config.num_objects, z)]
        ).astype(np.float32)

        self.destination_positions = np.column_stack(
            [destination_xy, np.full(self.config.num_destinations, z)]
        ).astype(np.float32)

        self.goal_object_idx = int(
            self.np_random.integers(self.config.num_objects)
        )

        self.goal_destination_idx = int(
            self.np_random.integers(self.config.num_destinations)
        )

        self.goal = self.destination_positions[
            self.goal_destination_idx
        ].copy()

        identity = np.array([0.0, 0.0, 0.0, 1.0])

        for name, position in zip(
            self.object_names, self.object_positions
        ):
            self.sim.set_base_pose(name, position, identity)

        for name, position in zip(
            self.destination_names, self.destination_positions
        ):
            marker_position = position.copy()
            marker_position[2] = 0.001

            self.sim.set_base_pose(
                name, marker_position, identity
            )
            
        self.sim.physics_client.performCollisionDetection()

    def get_obs(self) -> np.ndarray:
        """Return object positions and selected object identity.

        Destination information is provided separately through desired_goal.
        This avoids inconsistency when HER relabels the desired goal.
        """
        object_positions = np.concatenate([
            self.sim.get_base_position(name)
            for name in self.object_names
        ])

        object_goal = np.zeros(
            self.config.num_objects, dtype=np.float32
        )
        object_goal[self.goal_object_idx] = 1.0

        return np.concatenate([
            object_positions,
            object_goal,
        ]).astype(np.float32)

    def get_achieved_goal(self) -> np.ndarray:
        return np.asarray(
            self.sim.get_base_position(
                self.object_names[self.goal_object_idx]
            ),
            dtype=np.float32,
        )

    def get_goal(self) -> np.ndarray:
        return self.goal.copy()

    def _get_selected_body_ids(self):
        """Return PyBullet IDs for the selected object and destination."""
        object_name = self.object_names[self.goal_object_idx]
        destination_name = self.destination_names[
            self.goal_destination_idx
        ]

        body_ids = self.sim._bodies_idx

        return (
            body_ids[object_name],
            body_ids[destination_name],
            body_ids["table"],
            body_ids["panda"],
        )

    def _is_object_inside_destination(
        self,
        physics,
        object_id: int,
        destination_id: int,
    ) -> bool:
        """Check whether the complete object AABB fits inside the target."""

        object_min, object_max = physics.getAABB(object_id)

        destination_center = np.asarray(
            physics.getBasePositionAndOrientation(
                destination_id
            )[0][:2]
        )

        destination_half_extents = np.asarray(
            self.config.destination_half_extents,
            dtype=np.float64,
        )

        margin = self.config.placement_margin

        target_min = destination_center - destination_half_extents + margin
        target_max = destination_center + destination_half_extents - margin

        object_min = np.asarray(object_min[:2])
        object_max = np.asarray(object_max[:2])

        return bool(
            np.all(object_min >= target_min)
            and np.all(object_max <= target_max)
        )

    def _is_object_supported(
        self,
        physics,
        object_id: int,
        table_id: int,
    ) -> bool:
        """Check for physical support from the table."""
        contacts = physics.getContactPoints(
            bodyA=object_id,
            bodyB=table_id,
        )

        return any(
            contact[9] > self.config.support_force_threshold
            for contact in contacts
        )

    def _is_gripper_released(
        self,
        physics,
        object_id: int,
        robot_id: int,
    ) -> bool:
        """Check that neither gripper finger touches the selected object."""
        contacts = physics.getContactPoints(
            bodyA=object_id,
            bodyB=robot_id,
        )

        finger_links = {9, 10}

        return not any(
            contact[4] in finger_links
            for contact in contacts
        )

    def _is_object_stationary(
        self,
        physics,
        object_id: int,
    ) -> bool:
        """Check that the object's linear and angular speeds are low."""
        linear_velocity, angular_velocity = (
            physics.getBaseVelocity(object_id)
        )

        linear_speed = np.linalg.norm(linear_velocity)
        angular_speed = np.linalg.norm(angular_velocity)

        return bool(
            linear_speed <= self.config.linear_velocity_threshold
            and angular_speed <= self.config.angular_velocity_threshold
        )

    def is_success(self, achieved_goal, desired_goal):
        """Require stable, physically valid placement for N steps."""
        physics = self.sim.physics_client

        (
            object_id,
            destination_id,
            table_id,
            robot_id,
        ) = self._get_selected_body_ids()

        conditions = (
            self._is_object_inside_destination(
                physics, object_id, destination_id
            ),
            self._is_object_supported(
                physics, object_id, table_id
            ),
            self._is_gripper_released(
                physics, object_id, robot_id
            ),
            self._is_object_stationary(
                physics, object_id
            ),
        )

        if all(conditions):
            self._success_streak += 1
        else:
            self._success_streak = 0

        return np.array(
            self._success_streak
            >= self.config.success_stability_steps,
            dtype=bool,
        )

    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        goal: np.ndarray,
        info: dict,
    ) -> np.ndarray:
        distance = np.linalg.norm(
            np.asarray(achieved_goal) - np.asarray(goal),
            axis=-1,
        )

        if self.config.reward_type == "dense":
            return -distance.astype(np.float32)

        return np.where(
            distance < self.config.distance_threshold,
            0.0,
            -1.0,
        ).astype(np.float32)


class MultiObjectPickAndPlaceEnv(RobotTaskEnv):
    def __init__(
        self,
        config: MultiObjectConfig | None = None,
        render_mode: str = "rgb_array",
        renderer: str = "Tiny",
    ):
        self.config = config or MultiObjectConfig()

        sim = PyBullet(
            render_mode=render_mode,
            renderer=renderer,
        )

        robot = Panda(
            sim,
            block_gripper=False,
            base_position=np.array([-0.6, 0.0, 0.0]),
            control_type="ee",
        )

        task = MultiObjectPickAndPlaceTask(
            sim=sim,
            config=self.config,
        )

        super().__init__(
            robot=robot,
            task=task,
            render_target_position=np.array([-0.3, 0.0, 0.0]),
            render_distance=1.4,
            render_yaw=45,
            render_pitch=-30,
        )
        
    def step(self, action):
        observation, reward, terminated, _, info = super().step(action)

        self.task._elapsed_steps += 1

        truncated = (
            self.task._elapsed_steps >= self.config.max_episode_steps
            and not terminated
        )

        return observation, reward, terminated, truncated, info