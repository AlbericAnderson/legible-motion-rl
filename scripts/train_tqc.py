from __future__ import annotations

import argparse
import csv
import json
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

from legible_motion.envs.multi_object import (
    MultiObjectConfig,
    MultiObjectPickAndPlaceEnv,
)


# ---------------------------------------------------------
# Argument parsing helpers
# ---------------------------------------------------------

def parse_network_architecture(value: str) -> list[int]:
    try:
        architecture = [int(x.strip()) for x in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Network architecture must be comma-separated integers."
        ) from exc

    if not architecture or any(x <= 0 for x in architecture):
        raise argparse.ArgumentTypeError(
            "All network layer sizes must be positive."
        )

    return architecture


def parse_entropy_coefficient(value: str):
    if value.lower() == "auto":
        return "auto"

    try:
        coefficient = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Entropy coefficient must be 'auto' or a positive number."
        ) from exc

    if coefficient <= 0:
        raise argparse.ArgumentTypeError(
            "Entropy coefficient must be positive."
        )

    return coefficient


def parse_args():
    parser = argparse.ArgumentParser(
        description="Configurable TQC training for multi-object manipulation."
    )

    # -----------------------------------------------------
    # Experiment
    # -----------------------------------------------------

    parser.add_argument(
        "--run-name",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=100_000,
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

    # -----------------------------------------------------
    # Environment / reward
    # -----------------------------------------------------

    parser.add_argument(
        "--reward-type",
        choices=["dense", "sparse"],
        default="dense",
    )

    # -----------------------------------------------------
    # HER
    # -----------------------------------------------------

    parser.add_argument(
        "--her",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable HER. Use --no-her to disable it.",
    )

    parser.add_argument(
        "--her-strategy",
        choices=["future", "final", "episode"],
        default="future",
    )

    parser.add_argument(
        "--her-sampled-goals",
        type=int,
        default=4,
    )

    # -----------------------------------------------------
    # TQC hyperparameters
    # -----------------------------------------------------

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--buffer-size",
        type=int,
        default=1_000_000,
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
        "--gamma",
        type=float,
        default=0.95,
    )

    parser.add_argument(
        "--tau",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--ent-coef",
        type=parse_entropy_coefficient,
        default="auto",
    )

    parser.add_argument(
        "--train-freq",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--gradient-steps",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--net-arch",
        type=parse_network_architecture,
        default=[512, 512, 512],
    )

    parser.add_argument(
        "--n-critics",
        type=int,
        default=2,
    )

    # -----------------------------------------------------
    # Observation normalization
    # -----------------------------------------------------

    parser.add_argument(
        "--normalize-obs",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument(
        "--normalize-reward",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    # -----------------------------------------------------
    # Evaluation / checkpointing
    # -----------------------------------------------------

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
        "--checkpoint-freq",
        type=int,
        default=25_000,
    )

    return parser.parse_args()


# ---------------------------------------------------------
# Environment factory
# ---------------------------------------------------------

def make_env(
    monitor_file: Path,
    reward_type: str,
):
    def _init():

        config = MultiObjectConfig(
            reward_type=reward_type,
        )

        env = MultiObjectPickAndPlaceEnv(
            config=config,
        )

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

    def __init__(
        self,
        eval_env,
        run_dir: Path,
        eval_freq: int,
        n_eval_episodes: int,
        verbose: int = 1,
    ):
        super().__init__(verbose)

        self.eval_env = eval_env
        self.run_dir = run_dir
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes

        self.best_success_rate = -1.0
        self.best_mean_reward = -np.inf

        self.metrics_file = run_dir / "evaluation_metrics.csv"

    def _on_step(self) -> bool:

        if self.n_calls % self.eval_freq != 0:
            return True

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
            deterministic=True,
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
        print(f"Physical success rate: {success_rate * 100:.1f}%")
        print("=" * 60)

        write_header = not self.metrics_file.exists()

        with self.metrics_file.open("a", newline="") as file:

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

    args = parse_args()

    if args.total_timesteps <= 0:
        raise ValueError("total-timesteps must be positive.")

    if args.learning_starts < 0:
        raise ValueError("learning-starts cannot be negative.")

    if args.batch_size <= 0 or args.buffer_size <= 0:
        raise ValueError("Batch size and buffer size must be positive.")

    if args.n_critics <= 0:
        raise ValueError("n-critics must be positive.")

    if args.eval_freq <= 0 or args.eval_episodes <= 0:
        raise ValueError("Evaluation frequency and episodes must be positive.")

    if args.checkpoint_freq <= 0:
        raise ValueError("Checkpoint frequency must be positive.")

    if args.her_sampled_goals <= 0:
        raise ValueError("HER sampled goals must be positive.")

    # -----------------------------------------------------
    # Directories
    # -----------------------------------------------------

    run_dir = Path("artifacts/tqc") / args.run_name

    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(
            f"Run directory already exists and is not empty: {run_dir}\n"
            "Choose a different --run-name to avoid overwriting an experiment."
        )

    run_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    tensorboard_dir = run_dir / "tensorboard"
    tensorboard_dir.mkdir(exist_ok=True)

    # -----------------------------------------------------
    # Replay buffer configuration
    # -----------------------------------------------------

    if args.her:

        replay_buffer_class = HerReplayBuffer

        replay_buffer_kwargs = {
            "n_sampled_goal": args.her_sampled_goals,
            "goal_selection_strategy": args.her_strategy,
        }

    else:

        # SB3 uses its standard ReplayBuffer when this is None.
        replay_buffer_class = None
        replay_buffer_kwargs = None

    # -----------------------------------------------------
    # Save experiment configuration
    # -----------------------------------------------------

    experiment_config = {
        "run_name": args.run_name,
        "seed": args.seed,
        "total_timesteps": args.total_timesteps,

        "environment": {
            "name": "MultiObjectPickAndPlaceEnv",
            "reward_type": args.reward_type,
        },

        "algorithm": {
            "name": "TQC",
            "learning_rate": args.learning_rate,
            "buffer_size": args.buffer_size,
            "learning_starts": args.learning_starts,
            "batch_size": args.batch_size,
            "gamma": args.gamma,
            "tau": args.tau,
            "ent_coef": args.ent_coef,
            "train_freq": args.train_freq,
            "gradient_steps": args.gradient_steps,
            "net_arch": args.net_arch,
            "n_critics": args.n_critics,
        },

        "her": {
            "enabled": args.her,
            "strategy": args.her_strategy if args.her else None,
            "n_sampled_goal": (
                args.her_sampled_goals if args.her else None
            ),
        },

        "normalization": {
            "norm_obs": args.normalize_obs,
            "norm_reward": args.normalize_reward,
        },

        "evaluation": {
            "frequency": args.eval_freq,
            "episodes": args.eval_episodes,
        },

        "checkpoint_frequency": args.checkpoint_freq,

        "device": args.device,
    }

    with (run_dir / "config.json").open("w") as file:
        json.dump(experiment_config, file, indent=4)

    # -----------------------------------------------------
    # Training environment
    # -----------------------------------------------------

    train_env = DummyVecEnv([
        make_env(
            run_dir / "training_monitor.csv",
            args.reward_type,
        )
    ])

    train_env.seed(args.seed)

    train_env = VecNormalize(
        train_env,
        norm_obs=args.normalize_obs,
        norm_reward=args.normalize_reward,
        gamma=args.gamma,
    )

    # -----------------------------------------------------
    # Evaluation environment
    # -----------------------------------------------------

    eval_env = DummyVecEnv([
        make_env(
            run_dir / "evaluation_monitor.csv",
            args.reward_type,
        )
    ])

    eval_env.seed(args.seed + 10_000)

    eval_env = VecNormalize(
        eval_env,
        norm_obs=args.normalize_obs,
        norm_reward=args.normalize_reward,
        gamma=args.gamma,
    )

    eval_env.training = False
    eval_env.norm_reward = False

    # -----------------------------------------------------
    # TQC
    # -----------------------------------------------------

    policy_kwargs = {
        "net_arch": args.net_arch,
        "n_critics": args.n_critics,
    }

    model = TQC(
        policy="MultiInputPolicy",
        env=train_env,

        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,

        tau=args.tau,
        gamma=args.gamma,

        ent_coef=args.ent_coef,

        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,

        replay_buffer_class=replay_buffer_class,
        replay_buffer_kwargs=replay_buffer_kwargs,

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
        save_freq=args.checkpoint_freq,
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
    print("=" * 60)
    print(f"Run directory: {run_dir}")
    print(f"Training steps: {args.total_timesteps}")
    print(f"Reward type: {args.reward_type}")
    print(f"HER enabled: {args.her}")

    if args.her:
        print(f"HER strategy: {args.her_strategy}")
        print(f"HER sampled goals: {args.her_sampled_goals}")

    print(f"Learning rate: {args.learning_rate}")
    print(f"Batch size: {args.batch_size}")
    print(f"Network: {args.net_arch}")
    print(f"Critics: {args.n_critics}")
    print(f"Device: {args.device}")
    print("=" * 60)

    try:

        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            log_interval=10,
            tb_log_name="TQC",
        )

        # -------------------------------------------------
        # Final checkpoint
        # -------------------------------------------------

        model.save(
            str(run_dir / "final_model")
        )

        train_env.save(
            str(run_dir / "vecnormalize.pkl")
        )

        print("\nTraining completed.")
        print(f"Final model: {run_dir / 'final_model.zip'}")
        print(f"Normalization: {run_dir / 'vecnormalize.pkl'}")

    finally:

        train_env.close()
        eval_env.close()


if __name__ == "__main__":
    main()