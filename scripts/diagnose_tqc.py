import argparse
import csv
from pathlib import Path

import gymnasium as gym
import numpy as np

from sb3_contrib import TQC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from legible_motion.envs.multi_object import MultiObjectPickAndPlaceEnv


def scalar_bool(value):
    """Convert scalar or one-element array values to bool."""
    return bool(np.asarray(value).reshape(-1)[0])


def get_success(env, info):
    if "is_success" in info:
        return scalar_bool(info["is_success"])

    return scalar_bool(env.task.is_success())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("artifacts/tqc/corrected_obs_100k"),
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    model_path = args.run_dir / "best_model.zip"
    norm_path = args.run_dir / "best_vecnormalize.pkl"

    if not model_path.exists():
        raise FileNotFoundError(model_path)
    if not norm_path.exists():
        raise FileNotFoundError(norm_path)

    output_dir = args.run_dir / "diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the observation-normalization statistics used during training.
    # This VecEnv is used only to restore VecNormalize; evaluation itself
    # runs on a raw Gymnasium environment so terminal physical states remain
    # accessible before the next reset.
    norm_venv = DummyVecEnv(
        [lambda: MultiObjectPickAndPlaceEnv()]
    )
    vec_normalize = VecNormalize.load(norm_path, norm_venv)
    vec_normalize.training = False
    vec_normalize.norm_reward = False

    model = TQC.load(
        model_path,
        env=vec_normalize,
        device=args.device,
    )

    env = MultiObjectPickAndPlaceEnv()

    step_rows = []
    episode_rows = []

    try:
        for episode in range(args.episodes):
            obs, reset_info = env.reset(seed=args.seed + episode)

            task = env.task
            physics = env.sim.physics_client

            object_idx = task.goal_object_idx
            destination_idx = task.goal_destination_idx

            object_name = task.object_names[object_idx]
            destination_name = task.destination_names[destination_idx]

            object_id, destination_id, table_id, robot_id = (
                task._get_selected_body_ids()
            )

            target_position = np.asarray(task.goal, dtype=float).copy()
            initial_object_position = np.asarray(
                env.sim.get_base_position(object_name), dtype=float
            )

            max_object_height = float(initial_object_position[2])
            max_object_displacement = 0.0
            min_ee_object_distance = float("inf")
            min_object_target_distance = float("inf")

            any_finger_contact = False
            bilateral_contact_steps = 0
            max_finger_contact_count = 0

            episode_success = False
            episode_length = 0

            print(
                f"\nEpisode {episode + 1}/{args.episodes}: "
                f"{object_name} -> {destination_name}"
            )

            while True:
                # Normalize the raw observation with the saved training stats.
                normalized_obs = vec_normalize.normalize_obs(obs)

                action, _ = model.predict(
                    normalized_obs,
                    deterministic=True,
                )
                action = np.asarray(action, dtype=float).reshape(-1)

                obs, reward, terminated, truncated, info = env.step(action)
                episode_length += 1

                # Read the physical state AFTER the action has been applied.
                ee_position = np.asarray(
                    env.robot.get_ee_position(), dtype=float
                )
                finger_width = float(env.robot.get_fingers_width())

                object_position = np.asarray(
                    env.sim.get_base_position(object_name), dtype=float
                )

                ee_object_distance = float(
                    np.linalg.norm(ee_position - object_position)
                )
                object_target_distance = float(
                    np.linalg.norm(object_position - target_position)
                )

                object_velocity, object_angular_velocity = (
                    physics.getBaseVelocity(object_id)
                )

                object_speed = float(np.linalg.norm(object_velocity))
                object_angular_speed = float(
                    np.linalg.norm(object_angular_velocity)
                )

                # Object-to-robot contacts. PyBullet contact tuple index 4
                # is the robot-side link index in this query.
                robot_contacts = physics.getContactPoints(
                    bodyA=object_id,
                    bodyB=robot_id,
                )

                finger_links = {
                    contact[4]
                    for contact in robot_contacts
                    if contact[4] in (9, 10)
                }

                finger_contact_count = len(finger_links)
                finger_contact = finger_contact_count > 0
                bilateral_contact = finger_links == {9, 10}

                any_finger_contact |= finger_contact
                bilateral_contact_steps += int(bilateral_contact)
                max_finger_contact_count = max(
                    max_finger_contact_count,
                    finger_contact_count,
                )

                # Object-to-table contacts and their normal forces.
                table_contacts = physics.getContactPoints(
                    bodyA=object_id,
                    bodyB=table_id,
                )
                table_normal_force = sum(
                    float(contact[9]) for contact in table_contacts
                )

                max_object_height = max(
                    max_object_height,
                    float(object_position[2]),
                )

                object_displacement = float(
                    np.linalg.norm(
                        object_position - initial_object_position
                    )
                )
                max_object_displacement = max(
                    max_object_displacement,
                    object_displacement,
                )

                min_ee_object_distance = min(
                    min_ee_object_distance,
                    ee_object_distance,
                )
                min_object_target_distance = min(
                    min_object_target_distance,
                    object_target_distance,
                )

                episode_success = get_success(env, info)

                row = {
                    "episode": episode + 1,
                    "step": episode_length,
                    "object_name": object_name,
                    "destination_name": destination_name,
                    "ee_x": ee_position[0],
                    "ee_y": ee_position[1],
                    "ee_z": ee_position[2],
                    "object_x": object_position[0],
                    "object_y": object_position[1],
                    "object_z": object_position[2],
                    "target_x": target_position[0],
                    "target_y": target_position[1],
                    "target_z": target_position[2],
                    "ee_object_distance": ee_object_distance,
                    "object_target_distance": object_target_distance,
                    "finger_width": finger_width,
                    "object_speed": object_speed,
                    "object_angular_speed": object_angular_speed,
                    "object_displacement": object_displacement,
                    "finger_contact_count": finger_contact_count,
                    "bilateral_finger_contact": bilateral_contact,
                    "table_normal_force": table_normal_force,
                    "reward": float(reward),
                    "action_0": action[0],
                    "action_1": action[1],
                    "action_2": action[2],
                    "action_3": action[3],
                    "success": episode_success,
                }
                step_rows.append(row)

                if terminated or truncated:
                    break

            episode_row = {
                "episode": episode + 1,
                "object_name": object_name,
                "destination_name": destination_name,
                "episode_length": episode_length,
                "success": episode_success,
                "initial_object_x": initial_object_position[0],
                "initial_object_y": initial_object_position[1],
                "initial_object_z": initial_object_position[2],
                "final_object_x": object_position[0],
                "final_object_y": object_position[1],
                "final_object_z": object_position[2],
                "max_object_height": max_object_height,
                "max_object_displacement": max_object_displacement,
                "min_ee_object_distance": min_ee_object_distance,
                "min_object_target_distance": min_object_target_distance,
                "any_finger_contact": any_finger_contact,
                "bilateral_contact_steps": bilateral_contact_steps,
                "max_finger_contact_count": max_finger_contact_count,
                "final_finger_width": finger_width,
            }
            episode_rows.append(episode_row)

            print(
                f"  Success: {episode_success} | "
                f"Steps: {episode_length} | "
                f"Max object height: {max_object_height:.4f} m | "
                f"Max displacement: {max_object_displacement:.4f} m | "
                f"Min EE-object distance: {min_ee_object_distance:.4f} m | "
                f"Min object-target distance: "
                f"{min_object_target_distance:.4f} m | "
                f"Finger contact: {any_finger_contact} | "
                f"Bilateral contact steps: {bilateral_contact_steps}"
            )

    finally:
        env.close()
        norm_venv.close()

    # Write per-step physical trajectory data.
    step_csv = output_dir / "steps.csv"
    with step_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=step_rows[0].keys())
        writer.writeheader()
        writer.writerows(step_rows)

    # Write one summary row per episode.
    episode_csv = output_dir / "episodes.csv"
    with episode_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=episode_rows[0].keys())
        writer.writeheader()
        writer.writerows(episode_rows)

    success_rate = np.mean(
        [row["success"] for row in episode_rows]
    )

    print("\n=== Diagnostic summary ===")
    print(f"Episodes: {len(episode_rows)}")
    print(f"Success rate: {success_rate:.1%}")
    print(f"Step data: {step_csv}")
    print(f"Episode summaries: {episode_csv}")


if __name__ == "__main__":
    main()
