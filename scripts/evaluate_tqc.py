from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from sb3_contrib import TQC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecNormalize,
)

from legible_motion.envs.multi_object import MultiObjectPickAndPlaceEnv


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--vecnormalize",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=50,
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

    args = parser.parse_args()

    model_path = Path(args.model)
    vecnormalize_path = Path(args.vecnormalize)

    # -----------------------------------------------------
    # Environment
    # -----------------------------------------------------

    def make_env():
        return Monitor(
            MultiObjectPickAndPlaceEnv(),
            info_keywords=("is_success",),
        )

    raw_env = DummyVecEnv([make_env])

    env = VecNormalize.load(
        str(vecnormalize_path),
        raw_env,
    )

    env.training = False
    env.norm_reward = False
    env.seed(args.seed)

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------

    model = TQC.load(
        str(model_path),
        env=env,
        device=args.device,
    )

    # -----------------------------------------------------
    # Evaluation
    # -----------------------------------------------------

    episode_rewards = []
    episode_lengths = []
    episode_successes = []

    obs = env.reset()

    for episode in range(args.episodes):

        done = False
        total_reward = 0.0
        episode_length = 0

        while not done:

            action, _ = model.predict(
                obs,
                deterministic=True,
            )

            obs, rewards, dones, infos = env.step(action)

            total_reward += float(rewards[0])
            episode_length += 1

            done = bool(dones[0])

        info = infos[0]

        success = bool(
            np.asarray(
                info.get("is_success", False)
            ).item()
        )

        episode_rewards.append(total_reward)
        episode_lengths.append(episode_length)
        episode_successes.append(float(success))

        print(
            f"Episode {episode + 1:03d}/{args.episodes}: "
            f"reward={total_reward:.3f}, "
            f"length={episode_length}, "
            f"success={success}"
        )

    # -----------------------------------------------------
    # Results
    # -----------------------------------------------------

    mean_reward = float(np.mean(episode_rewards))
    std_reward = float(np.std(episode_rewards))

    mean_length = float(np.mean(episode_lengths))
    success_rate = float(np.mean(episode_successes))

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    print(f"Episodes: {args.episodes}")
    print(f"Mean reward: {mean_reward:.3f} +/- {std_reward:.3f}")
    print(f"Mean episode length: {mean_length:.1f}")
    print(f"Success rate: {success_rate * 100:.1f}%")

    print("=" * 60)

    # -----------------------------------------------------
    # Save results
    # -----------------------------------------------------

    output_file = model_path.parent / "evaluation_results.csv"

    with output_file.open(
        "w",
        newline="",
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "episode",
            "reward",
            "length",
            "success",
        ])

        for i in range(args.episodes):

            writer.writerow([
                i + 1,
                episode_rewards[i],
                episode_lengths[i],
                episode_successes[i],
            ])

    print(f"Results saved to: {output_file}")

    env.close()


if __name__ == "__main__":
    main()