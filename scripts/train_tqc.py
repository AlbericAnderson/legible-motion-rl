from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from sb3_contrib import TQC
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
)
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecNormalize,
    sync_envs_normalization,
)
from stable_baselines3.her import HerReplayBuffer

from legible_motion.envs.multi_object import MultiObjectPickAndPlaceEnv


# ---------------------------------------------------------
# Benchmark configuration
# ---------------------------------------------------------

LEARNING_RATE = 1e-3
BUFFER_SIZE = 1_000_000
GAMMA = 0.95
TAU = 0.05

NETWORK_ARCHITECTURE = [512, 512, 512]
N_CRITICS = 2

HER_CONFIG = {
    "n_sampled_goal": 4,
    "goal_selection_strategy": "future",
}


# ---------------------------------------------------------
# Environment factory
# ---------------------------------------------------------

def make_env(monitor_file: Path):
    def _init():
        env = MultiObjectPickAndPlaceEnv()

        env = Monitor(
            env,
            filename=str(monitor_file),
            info_keywords=("is_success",),
        )

        return env

    return _init


# ---------------------------------------------------------
# Evaluation callback
# ---------------------------------------------------------

class SuccessRateEvalCallback(BaseCallback):
    """
    Evaluate the policy periodically.

    Track:
      - Mean episode reward
      - Mean episode length
      - Physical success rate

    Save the best model according to success rate,
    using mean reward as a tie-breaker.
    """

    def __init__(
        self,
        eval_env,
        run_dir: Path,
        eval_freq: int = 10_000,
        n_eval_episodes: int = 10,
        deterministic: bool = True,
        verbose: int = 1,
    ):
        super().__init__(verbose)

        self.eval_env = eval_env
        self.run_dir = run_dir
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic

        self.best_success_rate = -1.0
        self.best_mean_reward = -np.inf

        self.metrics_file = run_dir / "evaluation_metrics.csv"

    def _on_step(self) -> bool:

        if self.n_calls % self.eval_freq != 0:
            return True

        # Synchronize observation-normalization statistics.
        sync_envs_normalization(
            self.training_env,
            self.eval_env,
        )

        self.eval_env.training = False
        self.eval_env.norm_reward = False

        successes = []

        def collect_success(locals_, globals_):
            if locals_.get("done", False):
                info = locals_.get("info", {})

                success = info.get("is_success", False)

                successes.append(
                    float(np.asarray(success).item())
                )

        episode_rewards, episode_lengths = evaluate_policy(
            self.model,
            self.eval_env,
            n_eval_episodes=self.n_eval_episodes,
            deterministic=self.deterministic,
            callback=collect_success,
            return_episode_rewards=True,
            warn=False,
        )

        mean_reward = float(np.mean(episode_rewards))
        std_reward = float(np.std(episode_rewards))

        mean_length = float(np.mean(episode_lengths))

        success_rate = (
            float(np.mean(successes))
            if successes
            else 0.0
        )

        self.logger.record("eval/mean_reward", mean_reward)
        self.logger.record("eval/std_reward", std_reward)
        self.logger.record("eval/mean_episode_length", mean_length)
        self.logger.record("eval/success_rate", success_rate)

        print("\n" + "=" * 60)
        print(f"Evaluation at timestep: {self.num_timesteps}")
        print(f"Mean reward: {mean_reward:.3f} +/- {std_reward:.3f}")
        print(f"Mean episode length: {mean_length:.1f}")
        print(
            f"Physical success rate: "
            f"{success_rate * 100:.1f}% "
            f"({len(successes)} episodes)"
        )
        print("=" * 60)

        # Append evaluation metrics.
        write_header = not self.metrics_file.exists()

        with self.metrics_file.open(
            "a",
            newline="",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "timesteps",
                    "mean_reward",
                    "std_reward",
                    "mean_episode_length",
                    "success_rate",
                ],
            )

            if write_header:
                writer.writeheader()

            writer.writerow({
                "timesteps": self.num_timesteps,
                "mean_reward": mean_reward,
                "std_reward": std_reward,
                "mean_episode_length": mean_length,
                "success_rate": success_rate,
            })

        # Save the best model.
        is_better = (
            success_rate > self.best_success_rate
            or (
                success_rate == self.best_success_rate
                and mean_reward > self.best_mean_reward
            )
        )

        if is_better:

            self.best_success_rate = success_rate
            self.best_mean_reward = mean_reward

            self.model.save(
                str(self.run_dir / "best_model")
            )

            self.training_env.save(
                str(self.run_dir / "best_vecnormalize.pkl")
            )

            print("Saved new best model.")

        return True


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=100_000,
    )

    parser.add_argument(
        "--learning-starts",
        type=int,
        default=1_000,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=2048,
    )

    parser.add_argument(
        "--eval-freq",
        type=int,
        default=10_000,
    )

    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default="tqc_pilot",
    )

    args = parser.parse_args()

    # -----------------------------------------------------
    # Directories
    # -----------------------------------------------------

    run_dir = Path("artifacts/tqc") / args.run_name

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    tensorboard_dir = run_dir / "tensorboard"
    tensorboard_dir.mkdir(exist_ok=True)

    # -----------------------------------------------------
    # Training environment
    # -----------------------------------------------------

    train_env = DummyVecEnv([
        make_env(run_dir / "training_monitor.csv")
    ])

    train_env.seed(args.seed)

    train_env = VecNormalize(
        train_env,
        norm_obs=True,
        norm_reward=False,
        gamma=GAMMA,
    )

    # -----------------------------------------------------
    # Evaluation environment
    # -----------------------------------------------------

    eval_env = DummyVecEnv([
        make_env(run_dir / "evaluation_monitor.csv")
    ])

    eval_env.seed(args.seed + 10_000)

    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,
        gamma=GAMMA,
    )

    eval_env.training = False
    eval_env.norm_reward = False

    # -----------------------------------------------------
    # TQC
    # -----------------------------------------------------

    policy_kwargs = {
        "net_arch": NETWORK_ARCHITECTURE,
        "n_critics": N_CRITICS,
    }

    model = TQC(
        policy="MultiInputPolicy",
        env=train_env,

        learning_rate=LEARNING_RATE,
        buffer_size=BUFFER_SIZE,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,

        tau=TAU,
        gamma=GAMMA,

        ent_coef="auto",

        train_freq=1,
        gradient_steps=1,

        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=HER_CONFIG,

        policy_kwargs=policy_kwargs,

        tensorboard_log=str(tensorboard_dir),

        verbose=1,
        seed=args.seed,
        device=args.device,
    )

    # -----------------------------------------------------
    # Callbacks
    # -----------------------------------------------------

    checkpoint_callback = CheckpointCallback(
        save_freq=25_000,
        save_path=str(checkpoint_dir),
        name_prefix="tqc",
        save_vecnormalize=True,
        save_replay_buffer=False,
        verbose=1,
    )

    evaluation_callback = SuccessRateEvalCallback(
        eval_env=eval_env,
        run_dir=run_dir,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        verbose=1,
    )

    callbacks = CallbackList([
        checkpoint_callback,
        evaluation_callback,
    ])

    # -----------------------------------------------------
    # Training
    # -----------------------------------------------------

    print("\nStarting TQC training")
    print(f"Run directory: {run_dir}")
    print(f"Training steps: {args.total_timesteps}")
    print(f"Learning starts: {args.learning_starts}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {args.device}")

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        log_interval=10,
        tb_log_name="TQC",
    )

    # -----------------------------------------------------
    # Final checkpoint
    # -----------------------------------------------------

    model.save(
        str(run_dir / "final_model")
    )

    train_env.save(
        str(run_dir / "vecnormalize.pkl")
    )

    print("\nTraining completed.")
    print(f"Final model: {run_dir / 'final_model.zip'}")
    print(f"Normalization: {run_dir / 'vecnormalize.pkl'}")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()