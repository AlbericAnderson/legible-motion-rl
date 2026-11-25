import argparse
import csv
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import TQC

from legible_motion.envs.multi_object import MultiObjectPickAndPlaceEnv


def make_env():
    return Monitor(
        MultiObjectPickAndPlaceEnv(
            render_mode="rgb_array",
            renderer="Tiny",
        )
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default="artifacts/tqc/tqc_pilot/final_model.zip",
    )
    parser.add_argument(
        "--vecnormalize",
        default="artifacts/tqc/tqc_pilot/vecnormalize.pkl",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="artifacts/replays")

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dummy_env = DummyVecEnv([make_env])

    env = VecNormalize.load(
        args.vecnormalize,
        dummy_env,
    )

    env.training = False
    env.norm_reward = False

    model = TQC.load(
        args.model,
        env=env,
        device="auto",
    )

    raw_env = env.venv.envs[0].unwrapped

    physics = raw_env.sim.physics_client
    robot_id = raw_env.sim._bodies_idx["panda"]

    for deterministic in [True, False]:

        mode = "deterministic" if deterministic else "stochastic"

        print(f"\nGenerating {mode} replay...")

        env.seed(args.seed)
        obs = env.reset()

        frames = []
        records = []

        # Capture the initial frame.
        frames.append(env.get_images()[0])

        for step in range(raw_env.config.max_episode_steps):

            action, _ = model.predict(
                obs,
                deterministic=deterministic,
            )

            obs, rewards, dones, infos = env.step(action)

            object_position = raw_env.task.get_achieved_goal()
            goal_position = raw_env.task.get_goal()

            distance = np.linalg.norm(
                object_position - goal_position
            )

            ee_state = physics.getLinkState(
                robot_id,
                11,
                computeLinkVelocity=1,
            )

            ee_position = np.asarray(ee_state[4])

            frames.append(env.get_images()[0])

            records.append({
                "step": step + 1,
                "reward": float(rewards[0]),
                "distance": float(distance),
                "object_x": float(object_position[0]),
                "object_y": float(object_position[1]),
                "object_z": float(object_position[2]),
                "goal_x": float(goal_position[0]),
                "goal_y": float(goal_position[1]),
                "goal_z": float(goal_position[2]),
                "ee_x": float(ee_position[0]),
                "ee_y": float(ee_position[1]),
                "ee_z": float(ee_position[2]),
                "action_x": float(action[0][0]),
                "action_y": float(action[0][1]),
                "action_z": float(action[0][2]),
                "gripper_action": float(action[0][3]),
                "terminated": bool(dones[0]),
                "success": bool(infos[0].get("is_success", False)),
            })

            if dones[0]:
                break

        video_path = output_dir / f"{mode}.mp4"

        imageio.mimsave(
            video_path,
            frames,
            fps=25,
        )

        csv_path = output_dir / f"{mode}.csv"

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=records[0].keys(),
            )
            writer.writeheader()
            writer.writerows(records)

        print("Video:", video_path)
        print("Data:", csv_path)
        print("Steps:", len(records))
        print("Final distance:", records[-1]["distance"])
        print("Success:", records[-1]["success"])

    env.close()


if __name__ == "__main__":
    main()