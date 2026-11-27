from __future__ import annotations

import argparse
import csv
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from sb3_contrib import TQC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from legible_motion.envs.multi_object import (
    MultiObjectConfig,
    MultiObjectPickAndPlaceEnv,
)


def make_env():
    return Monitor(
        MultiObjectPickAndPlaceEnv(
            config=MultiObjectConfig(reward_type="dense"),
            render_mode="rgb_array",
            renderer="Tiny",
        )
    )


def as_float(value):
    return float(np.asarray(value).item())


def get_physical_state(raw_env):
    """
    Evaluate the original strict physical placement conditions.

    Returns:
        Dictionary containing the individual physical conditions.
    """

    task = raw_env.task
    physics = raw_env.sim.physics_client

    object_id, destination_id, table_id, robot_id = (
        task._get_selected_body_ids()
    )

    # -----------------------------------------------------
    # Object containment inside destination
    # -----------------------------------------------------

    object_aabb = physics.getAABB(object_id)

    object_min = np.asarray(object_aabb[0])
    object_max = np.asarray(object_aabb[1])

    goal = np.asarray(task.get_goal())

    half_x, half_y = raw_env.config.destination_half_extents

    placement_margin = raw_env.config.placement_margin

    inside_destination = (
        object_min[0] >= goal[0] - half_x + placement_margin
        and object_max[0] <= goal[0] + half_x - placement_margin
        and object_min[1] >= goal[1] - half_y + placement_margin
        and object_max[1] <= goal[1] + half_y - placement_margin
    )

    # -----------------------------------------------------
    # Object-table support force
    # -----------------------------------------------------

    table_contacts = physics.getContactPoints(
        bodyA=object_id,
        bodyB=table_id,
    )

    support_force = sum(
        max(0.0, float(contact[9]))
        for contact in table_contacts
    )

    supported = (
        support_force >= raw_env.config.support_force_threshold
    )

    # -----------------------------------------------------
    # Finger contacts
    # -----------------------------------------------------

    robot_contacts = physics.getContactPoints(
        bodyA=object_id,
        bodyB=robot_id,
    )

    finger_contact_links = {9, 10}

    finger_contacts = [
        contact
        for contact in robot_contacts
        if contact[4] in finger_contact_links
    ]

    no_finger_contact = len(finger_contacts) == 0

    # -----------------------------------------------------
    # Object velocity
    # -----------------------------------------------------

    linear_velocity, angular_velocity = physics.getBaseVelocity(
        object_id
    )

    linear_speed = float(
        np.linalg.norm(linear_velocity)
    )

    angular_speed = float(
        np.linalg.norm(angular_velocity)
    )

    stationary = (
        linear_speed <= raw_env.config.linear_velocity_threshold
        and angular_speed <= raw_env.config.angular_velocity_threshold
    )

    return {
        "inside_destination": bool(inside_destination),
        "supported": bool(supported),
        "no_finger_contact": bool(no_finger_contact),
        "stationary": bool(stationary),
        "support_force": float(support_force),
        "linear_speed": linear_speed,
        "angular_speed": angular_speed,
    }


def normalize_observation(vec_env, observation):
    """
    Apply the saved VecNormalize statistics to one raw observation.
    """

    batched_observation = {
        key: np.expand_dims(value, axis=0)
        for key, value in observation.items()
    }

    return vec_env.normalize_obs(batched_observation)


def evaluate_checkpoint(
    model_path: Path,
    vecnormalize_path: Path,
    output_dir: Path,
    episodes: int,
    seed: int,
    device: str,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print(f"CHECKPOINT: {model_path}")
    print(f"OUTPUT: {output_dir}")
    print("=" * 70)

    # -----------------------------------------------------
    # Environment and normalization
    # -----------------------------------------------------

    dummy_env = DummyVecEnv([make_env])

    vec_env = VecNormalize.load(
        str(vecnormalize_path),
        dummy_env,
    )

    vec_env.training = False
    vec_env.norm_reward = False

    raw_env = vec_env.venv.envs[0].unwrapped

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------

    model = TQC.load(
        str(model_path),
        env=vec_env,
        device=device,
    )

    # -----------------------------------------------------
    # Preserve original termination function
    # -----------------------------------------------------

    original_is_success = raw_env.task.is_success

    # Continue the diagnostic episode after reaching the
    # distance threshold. We calculate distance success
    # independently below.
    raw_env.task.is_success = lambda achieved_goal, desired_goal: (
        np.zeros(
            np.asarray(achieved_goal).shape[:-1],
            dtype=bool,
        )
    )

    # -----------------------------------------------------
    # Results
    # -----------------------------------------------------

    episode_summaries = []

    for episode in range(episodes):

        episode_seed = seed + episode

        print(
            f"\nEpisode {episode + 1:03d}/{episodes} "
            f"(seed={episode_seed})"
        )

        observation, _ = raw_env.reset(seed=episode_seed)

        frames = []
        step_records = []

        # Initial frame
        frame = raw_env.render()

        if frame is not None:
            frames.append(frame)

        total_reward = 0.0

        distance_success = False
        physical_success = False

        stable_steps = 0

        min_ee_object_distance = float("inf")
        min_object_target_distance = float("inf")

        initial_object_position = np.asarray(
            raw_env.task.get_achieved_goal()
        ).copy()

        max_object_displacement = 0.0

        episode_length = 0

        # -------------------------------------------------
        # Episode
        # -------------------------------------------------

        for step in range(raw_env.config.max_episode_steps):

            normalized_obs = normalize_observation(
                vec_env,
                observation,
            )

            action, _ = model.predict(
                normalized_obs,
                deterministic=True,
            )

            action = np.asarray(action[0], dtype=np.float32)

            observation, reward, terminated, truncated, info = (
                raw_env.step(action)
            )

            episode_length += 1
            total_reward += float(reward)

            # ---------------------------------------------
            # Object and target
            # ---------------------------------------------

            object_position = np.asarray(
                raw_env.task.get_achieved_goal()
            )

            goal_position = np.asarray(
                raw_env.task.get_goal()
            )

            object_target_distance = float(
                np.linalg.norm(object_position - goal_position)
            )

            min_object_target_distance = min(
                min_object_target_distance,
                object_target_distance,
            )

            if object_target_distance < raw_env.config.distance_threshold:
                distance_success = True

            displacement = float(
                np.linalg.norm(
                    object_position - initial_object_position
                )
            )

            max_object_displacement = max(
                max_object_displacement,
                displacement,
            )

            # ---------------------------------------------
            # Robot end-effector
            # ---------------------------------------------

            physics = raw_env.sim.physics_client

            robot_id = raw_env.sim._bodies_idx["panda"]

            ee_state = physics.getLinkState(
                robot_id,
                11,
                computeLinkVelocity=1,
            )

            ee_position = np.asarray(ee_state[4])

            ee_object_distance = float(
                np.linalg.norm(
                    ee_position - object_position
                )
            )

            min_ee_object_distance = min(
                min_ee_object_distance,
                ee_object_distance,
            )

            # ---------------------------------------------
            # Gripper
            # ---------------------------------------------

            finger_width = float(
                raw_env.robot.get_fingers_width()
            )

            # ---------------------------------------------
            # Strict physical placement
            # ---------------------------------------------

            physical_state = get_physical_state(raw_env)

            strict_conditions = (
                physical_state["inside_destination"]
                and physical_state["supported"]
                and physical_state["no_finger_contact"]
                and physical_state["stationary"]
            )

            if strict_conditions:
                stable_steps += 1
            else:
                stable_steps = 0

            if (
                stable_steps
                >= raw_env.config.success_stability_steps
            ):
                physical_success = True

            # ---------------------------------------------
            # Record frame
            # ---------------------------------------------

            frame = raw_env.render()

            if frame is not None:
                frames.append(frame)

            # ---------------------------------------------
            # Step diagnostics
            # ---------------------------------------------

            step_records.append({
                "episode": episode + 1,
                "step": step + 1,
                "reward": float(reward),

                "ee_x": float(ee_position[0]),
                "ee_y": float(ee_position[1]),
                "ee_z": float(ee_position[2]),

                "object_x": float(object_position[0]),
                "object_y": float(object_position[1]),
                "object_z": float(object_position[2]),

                "goal_x": float(goal_position[0]),
                "goal_y": float(goal_position[1]),
                "goal_z": float(goal_position[2]),

                "ee_object_distance": ee_object_distance,
                "object_target_distance": object_target_distance,

                "finger_width": finger_width,

                "action_x": float(action[0]),
                "action_y": float(action[1]),
                "action_z": float(action[2]),
                "gripper_action": float(action[3]),

                "inside_destination": physical_state["inside_destination"],
                "supported": physical_state["supported"],
                "no_finger_contact": physical_state["no_finger_contact"],
                "stationary": physical_state["stationary"],

                "support_force": physical_state["support_force"],
                "linear_speed": physical_state["linear_speed"],
                "angular_speed": physical_state["angular_speed"],

                "stable_steps": stable_steps,

                "distance_success": distance_success,
                "physical_success": physical_success,
            })

            if terminated or truncated:
                break

        # -------------------------------------------------
        # Save video
        # -------------------------------------------------

        video_path = output_dir / f"episode_{episode + 1:03d}.mp4"

        if frames:
            imageio.mimsave(
                video_path,
                frames,
                fps=25,
                macro_block_size=1,
            )

        # -------------------------------------------------
        # Save step diagnostics
        # -------------------------------------------------

        steps_path = (
            output_dir / f"episode_{episode + 1:03d}_steps.csv"
        )

        with steps_path.open("w", newline="") as file:

            writer = csv.DictWriter(
                file,
                fieldnames=step_records[0].keys(),
            )

            writer.writeheader()
            writer.writerows(step_records)

        # -------------------------------------------------
        # Episode summary
        # -------------------------------------------------

        summary = {
            "episode": episode + 1,
            "seed": episode_seed,
            "reward": total_reward,
            "length": episode_length,

            "distance_success": distance_success,
            "physical_success": physical_success,

            "min_ee_object_distance": min_ee_object_distance,
            "min_object_target_distance": min_object_target_distance,

            "max_object_displacement": max_object_displacement,

            "final_object_target_distance": object_target_distance,
        }

        episode_summaries.append(summary)

        print(
            f"Reward: {total_reward:.3f} | "
            f"Length: {episode_length} | "
            f"Min EE-object: {min_ee_object_distance:.3f} m | "
            f"Min object-target: {min_object_target_distance:.3f} m"
        )

        print(
            f"Distance success: {distance_success} | "
            f"Strict physical success: {physical_success}"
        )

        print(f"Video: {video_path}")

    # -----------------------------------------------------
    # Restore original success function
    # -----------------------------------------------------

    raw_env.task.is_success = original_is_success

    # -----------------------------------------------------
    # Save summary CSV
    # -----------------------------------------------------

    summary_path = output_dir / "episode_summary.csv"

    with summary_path.open("w", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=episode_summaries[0].keys(),
        )

        writer.writeheader()
        writer.writerows(episode_summaries)

    # -----------------------------------------------------
    # Aggregate metrics
    # -----------------------------------------------------

    distance_success_rate = np.mean([
        row["distance_success"]
        for row in episode_summaries
    ])

    physical_success_rate = np.mean([
        row["physical_success"]
        for row in episode_summaries
    ])

    mean_reward = np.mean([
        row["reward"]
        for row in episode_summaries
    ])

    mean_min_distance = np.mean([
        row["min_object_target_distance"]
        for row in episode_summaries
    ])

    print("\n" + "=" * 70)
    print("CHECKPOINT RESULTS")
    print("=" * 70)

    print(f"Episodes: {episodes}")
    print(f"Mean reward: {mean_reward:.3f}")

    print(
        f"Distance-based success: "
        f"{distance_success_rate * 100:.1f}%"
    )

    print(
        f"Strict physical success: "
        f"{physical_success_rate * 100:.1f}%"
    )

    print(
        f"Mean minimum object-target distance: "
        f"{mean_min_distance:.4f} m"
    )

    print(f"Summary: {summary_path}")
    print("=" * 70)

    vec_env.close()


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--best-model",
        type=str,
        default="artifacts/tqc/B_dense_no_her_distance_success/best_model.zip",
    )

    parser.add_argument(
        "--best-vecnormalize",
        type=str,
        default="artifacts/tqc/B_dense_no_her_distance_success/best_vecnormalize.pkl",
    )

    parser.add_argument(
        "--final-model",
        type=str,
        default="artifacts/tqc/B_dense_no_her_distance_success/final_model.zip",
    )

    parser.add_argument(
        "--final-vecnormalize",
        type=str,
        default="artifacts/tqc/B_dense_no_her_distance_success/vecnormalize.pkl",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/replays/B_dense_no_her_distance_success",
    )

    args = parser.parse_args()

    output_root = Path(args.output)

    checkpoints = [
        (
            "best",
            Path(args.best_model),
            Path(args.best_vecnormalize),
        ),
        (
            "final",
            Path(args.final_model),
            Path(args.final_vecnormalize),
        ),
    ]

    for name, model_path, vecnormalize_path in checkpoints:

        if not model_path.exists():
            raise FileNotFoundError(model_path)

        if not vecnormalize_path.exists():
            raise FileNotFoundError(vecnormalize_path)

        evaluate_checkpoint(
            model_path=model_path,
            vecnormalize_path=vecnormalize_path,
            output_dir=output_root / name,
            episodes=args.episodes,
            seed=args.seed,
            device=args.device,
        )


if __name__ == "__main__":
    main()