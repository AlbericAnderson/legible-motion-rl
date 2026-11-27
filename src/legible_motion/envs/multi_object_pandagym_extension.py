"""PandaGym PickAndPlace extended with distractor objects and destinations.

The original PandaGym PickAndPlace task remains responsible for the active
object, active target, observation components, reward, and success criterion.
Additional objects are static physical distractors; additional destinations
are visual-only markers.

Observation modes:
- target_only: exactly the original PandaGym task observation.
- multi_object_one_hot: original task observation + all logical object
  positions + one-hot identity of the selected logical object.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from panda_gym.envs.core import RobotTaskEnv
from panda_gym.envs.robots.panda import Panda
from panda_gym.envs.tasks.pick_and_place import PickAndPlace
from panda_gym.pybullet import PyBullet


ObservationMode = Literal["target_only", "multi_object_one_hot"]


@dataclass
class ExtendedPickAndPlaceConfig:
    num_objects: int = 2
    num_destinations: int = 2
    observation_mode: ObservationMode = "target_only"

    # Keep PandaGym's original sampling ranges by default.
    reward_type: str = "dense"
    distance_threshold: float = 0.05
    goal_xy_range: float = 0.3
    goal_z_range: float = 0.2
    obj_xy_range: float = 0.3

    # Minimum XY distance between every pair of object/destination centers.
    min_separation: float = 0.08

    # Episode time limit is a truncation, not an MDP terminal condition.
    max_episode_steps: int = 250

    def validate(self) -> None:
        if self.num_objects < 1:
            raise ValueError("num_objects must be at least 1.")
        if self.num_destinations < 1:
            raise ValueError("num_destinations must be at least 1.")
        if self.observation_mode not in {
            "target_only",
            "multi_object_one_hot",
        }:
            raise ValueError(
                "observation_mode must be 'target_only' or "
                "'multi_object_one_hot'."
            )
        if self.reward_type not in {"dense", "sparse"}:
            raise ValueError("reward_type must be 'dense' or 'sparse'.")
        if self.distance_threshold <= 0:
            raise ValueError("distance_threshold must be positive.")
        if self.min_separation < 0:
            raise ValueError("min_separation cannot be negative.")
        if self.max_episode_steps < 1:
            raise ValueError("max_episode_steps must be at least 1.")
        if self.goal_xy_range <= 0 or self.obj_xy_range <= 0:
            raise ValueError("XY sampling ranges must be positive.")


class ExtendedPickAndPlaceTask(PickAndPlace):
    """Original PandaGym task plus static clutter and independent goal choice."""

    def __init__(self, sim: PyBullet, config: ExtendedPickAndPlaceConfig):
        config.validate()
        self.config = config

        # This creates the original plane, table, dynamic "object", and
        # ghost "target", and initializes the original reward/success logic.
        super().__init__(
            sim=sim,
            reward_type=config.reward_type,
            distance_threshold=config.distance_threshold,
            goal_xy_range=config.goal_xy_range,
            goal_z_range=config.goal_z_range,
            obj_xy_range=config.obj_xy_range,
        )

        self.extra_object_names = [
            f"distractor_object_{i}"
            for i in range(config.num_objects - 1)
        ]
        self.extra_destination_names = [
            f"distractor_target_{i}"
            for i in range(config.num_destinations - 1)
        ]

        self._logical_object_names: list[str] = []
        self.selected_object_index = 0
        self.selected_destination_index = 0
        self._create_distractors()

    def step(self, action: np.ndarray):
        observation, reward, terminated, _truncated, info = super().step(
            action
        )

        self._elapsed_steps += 1

        truncated = (
            self._elapsed_steps >= self.config.max_episode_steps
            and not terminated
        )

        if self._elapsed_steps == 1:
            print(
                f"[EPISODE START] "
                f"object={self.task.selected_object_index}, "
                f"destination={self.task.selected_destination_index}"
            )

        if terminated or truncated:
            print(
                f"[EPISODE END] "
                f"steps={self._elapsed_steps}, "
                f"terminated={terminated}, "
                f"truncated={truncated}"
            )

        return observation, reward, terminated, truncated, info

    def _create_distractors(self) -> None:
        """Create additional bodies once; reset() only changes their poses."""
        object_colors = (
            np.array([0.95, 0.25, 0.10, 1.0]),
            np.array([0.10, 0.30, 0.95, 1.0]),
            np.array([0.90, 0.70, 0.10, 1.0]),
            np.array([0.65, 0.25, 0.85, 1.0]),
        )
        destination_colors = (
            np.array([0.95, 0.25, 0.10, 0.25]),
            np.array([0.10, 0.30, 0.95, 0.25]),
            np.array([0.90, 0.70, 0.10, 0.25]),
            np.array([0.65, 0.25, 0.85, 0.25]),
        )

        with self.sim.no_rendering():
            for i, name in enumerate(self.extra_object_names):
                self.sim.create_box(
                    body_name=name,
                    half_extents=np.ones(3) * self.object_size / 2,
                    # Static colliders: visible and physically solid, but
                    # they cannot be displaced by the robot.
                    mass=0.0,
                    position=np.array([0.0, 0.0, self.object_size / 2]),
                    rgba_color=object_colors[i % len(object_colors)],
                )

            for i, name in enumerate(self.extra_destination_names):
                self.sim.create_box(
                    body_name=name,
                    half_extents=np.ones(3) * self.object_size / 2,
                    mass=0.0,
                    ghost=True,
                    position=np.array([0.0, 0.0, self.object_size / 2]),
                    rgba_color=destination_colors[
                        i % len(destination_colors)
                    ],
                )

    def _sample_separated_xy(self, count: int) -> np.ndarray:
        """Sample object and destination XY positions in one shared workspace."""
        half_range = min(
            self.config.goal_xy_range,
            self.config.obj_xy_range,
        ) / 2.0
        low = np.array([-half_range, -half_range], dtype=np.float64)
        high = np.array([half_range, half_range], dtype=np.float64)

        points: list[np.ndarray] = []
        for _ in range(count):
            for _attempt in range(5000):
                candidate = self.np_random.uniform(low, high)
                if all(
                    np.linalg.norm(candidate - point)
                    >= self.config.min_separation
                    for point in points
                ):
                    points.append(candidate)
                    break
            else:
                raise RuntimeError(
                    "Could not sample separated scene positions. "
                    "Reduce min_separation or the number of bodies, "
                    "or increase the XY sampling ranges."
                )
        return np.asarray(points, dtype=np.float64)

    def _sample_destination_z(self) -> float:
        # Match PickAndPlace._sample_goal(): the goal cube center starts at
        # object_size/2, then receives the original vertical goal noise.
        z_noise = float(
            self.np_random.uniform(0.0, self.config.goal_z_range)
        )
        if self.np_random.random() < 0.3:
            z_noise = 0.0
        return self.object_size / 2.0 + z_noise

    def reset(self) -> None:
        """Choose an object and destination independently and place the scene."""
        self.selected_object_index = int(
            self.np_random.integers(self.config.num_objects)
        )
        self.selected_destination_index = int(
            self.np_random.integers(self.config.num_destinations)
        )

        total = self.config.num_objects + self.config.num_destinations
        xy = self._sample_separated_xy(total)
        object_xy = xy[: self.config.num_objects]
        destination_xy = xy[self.config.num_objects :]

        object_z = self.object_size / 2.0
        object_positions = np.column_stack(
            (
                object_xy,
                np.full(self.config.num_objects, object_z),
            )
        )

        destination_positions = np.column_stack(
            (
                destination_xy,
                np.asarray(
                    [
                        self._sample_destination_z()
                        for _ in range(self.config.num_destinations)
                    ]
                ),
            )
        )

        # The original "object" and "target" bodies represent whichever
        # logical object/destination was selected for this episode.
        identity = np.array([0.0, 0.0, 0.0, 1.0])
        selected_object_position = object_positions[
            self.selected_object_index
        ].copy()
        selected_destination_position = destination_positions[
            self.selected_destination_index
        ].copy()

        self.goal = selected_destination_position
        self.sim.set_base_pose(
            "object", selected_object_position, identity
        )
        self.sim.set_base_pose(
            "target", selected_destination_position, identity
        )

        # Map logical object indices to the canonical active body and the
        # additional static bodies. The mapping is fixed for this episode.
        available_extra_objects = iter(self.extra_object_names)
        self._logical_object_names = []
        for logical_index in range(self.config.num_objects):
            if logical_index == self.selected_object_index:
                self._logical_object_names.append("object")
            else:
                self._logical_object_names.append(
                    next(available_extra_objects)
                )

        for logical_index, body_name in enumerate(
            self._logical_object_names
        ):
            if logical_index == self.selected_object_index:
                continue
            self.sim.set_base_pose(
                body_name, object_positions[logical_index], identity
            )

        available_extra_targets = iter(self.extra_destination_names)
        for logical_index in range(self.config.num_destinations):
            if logical_index == self.selected_destination_index:
                continue
            body_name = next(available_extra_targets)
            self.sim.set_base_pose(
                body_name, destination_positions[logical_index], identity
            )

        self.sim.physics_client.performCollisionDetection()

    def get_obs(self) -> np.ndarray:
        """Return either the original task observation or its augmented form."""
        original_obs = super().get_obs().astype(np.float32)

        if self.config.observation_mode == "target_only":
            return original_obs

        all_object_positions = np.concatenate(
            [
                np.asarray(
                    self.sim.get_base_position(name),
                    dtype=np.float32,
                )
                for name in self._logical_object_names
            ]
        )
        object_identity = np.zeros(
            self.config.num_objects, dtype=np.float32
        )
        object_identity[self.selected_object_index] = 1.0

        return np.concatenate(
            [original_obs, all_object_positions, object_identity]
        ).astype(np.float32)


class MultiObjectPickAndPlaceEnv(RobotTaskEnv):
    """PandaGym PickAndPlace with independently selected cluttered scenes."""

    def __init__(
        self,
        config: ExtendedPickAndPlaceConfig | None = None,
        render_mode: str = "rgb_array",
        renderer: str = "Tiny",
        render_width: int = 720,
        render_height: int = 480,
    ):
        self.config = config or ExtendedPickAndPlaceConfig()
        self.config.validate()
        self._elapsed_steps = 0

        sim = PyBullet(render_mode=render_mode, renderer=renderer)
        robot = Panda(
            sim,
            block_gripper=False,
            base_position=np.array([-0.6, 0.0, 0.0]),
            control_type="ee",
        )
        task = ExtendedPickAndPlaceTask(sim=sim, config=self.config)

        super().__init__(
            robot=robot,
            task=task,
            render_width=render_width,
            render_height=render_height,
            render_target_position=np.array([-0.3, 0.0, 0.0]),
            render_distance=1.4,
            render_yaw=45,
            render_pitch=-30,
        )

    def reset(self, seed: int | None = None, options: dict | None = None):
        observation, info = super().reset(seed=seed, options=options)
        self._elapsed_steps = 0
        return observation, info

    def step(self, action: np.ndarray):
        observation, reward, terminated, _truncated, info = super().step(
            action
        )
        self._elapsed_steps += 1
        truncated = (
            self._elapsed_steps >= self.config.max_episode_steps
            and not terminated
        )
        return observation, reward, terminated, truncated, info
