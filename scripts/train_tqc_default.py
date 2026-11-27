from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import imageio.v2 as imageio

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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Standard TQC + HER training on the multi-object PandaGym environment."
    )

    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument(
        "--total-timesteps", type=int, default=500_000,
        help="Target cumulative timestep count (not additional steps when resuming).",
    )
    parser.add_argument(
        "--resume-from", type=Path, default=None,
        help="Model checkpoint .zip to resume, with matching HER replay-buffer and VecNormalize files.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")

    parser.add_argument(
        "--reward-type", choices=["dense", "sparse"], default="dense"
    )
    parser.add_argument("--max-episode-steps", type=int, default=60)

    # TQC hyperparameters: matched to the standard PandaGym baseline.
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--buffer-size", type=int, default=1_000_000)
    parser.add_argument("--learning-starts", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument("--ent-coef", type=str, default="auto")
    parser.add_argument("--train-freq", type=int, default=1)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument(
        "--net-arch", type=int, nargs="+", default=[512, 512, 512]
    )
    parser.add_argument("--n-critics", type=int, default=2)

    # HER.
    parser.add_argument("--her-sampled-goals", type=int, default=4)
    parser.add_argument(
        "--her-strategy",
        choices=["future", "final", "episode"],
        default="future",
    )

    # Observation/reward normalization.
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

    # Evaluation and checkpointing.
    parser.add_argument("--eval-freq", type=int, default=5_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--checkpoint-freq", type=int, default=25_000)

    args = parser.parse_args()

    if args.total_timesteps <= 0:
        parser.error("--total-timesteps must be positive")
    if args.max_episode_steps <= 0:
        parser.error("--max-episode-steps must be positive")
    if args.learning_starts < 0:
        parser.error("--learning-starts cannot be negative")
    if args.batch_size <= 0 or args.buffer_size <= 0:
        parser.error("--batch-size and --buffer-size must be positive")
    if args.eval_freq <= 0 or args.eval_episodes <= 0:
        parser.error("--eval-freq and --eval-episodes must be positive")
    if args.checkpoint_freq <= 0:
        parser.error("--checkpoint-freq must be positive")
    if args.her_sampled_goals <= 0 or args.n_critics <= 0:
        parser.error("--her-sampled-goals and --n-critics must be positive")

    return args


def make_env(
    monitor_path: Path,
    reward_type: str,
    max_episode_steps: int,
    render_mode: str = "rgb_array",
):
    def _init():
        config = MultiObjectConfig(
            reward_type=reward_type,
            max_episode_steps=max_episode_steps,
        )
        env = MultiObjectPickAndPlaceEnv(
            config=config,
            render_mode=render_mode,
        )
        return Monitor(
            env,
            filename=str(monitor_path),
            info_keywords=("is_success",),
        )

    return _init


def record_evaluation_video(model, eval_env, video_path: Path, max_steps: int):
    """Record one deterministic evaluation episode from the RGB-array eval env."""
    frames = []
    obs = eval_env.reset()

    frame = eval_env.render()
    if frame is not None:
        if isinstance(frame, (list, tuple)):
            frame = frame[0]
        frames.append(np.asarray(frame, dtype=np.uint8))

    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, _ = eval_env.step(action)

        # VecEnv auto-resets after a terminal step. Do not append that reset
        # scene as the final frame of the episode being recorded.
        if bool(np.asarray(dones).reshape(-1)[0]):
            break

        frame = eval_env.render()
        if frame is not None:
            if isinstance(frame, (list, tuple)):
                frame = frame[0]
            frames.append(np.asarray(frame, dtype=np.uint8))

    if not frames:
        raise RuntimeError(
            "The evaluation environment returned no RGB frames. "
            "Check that render_mode='rgb_array' is supported."
        )

    video_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(video_path, frames, fps=25, macro_block_size=1)
    print(f"Saved evaluation video: {video_path}")


class SuccessRateEvalCallback(BaseCallback):
    """Evaluate deterministic policy and save the best model/normalization."""

    def __init__(
        self,
        eval_env,
        run_dir: Path,
        eval_freq: int,
        n_eval_episodes: int,
        max_episode_steps: int,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.run_dir = run_dir
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.max_episode_steps = max_episode_steps
        self.best_success_rate = -1.0
        self.best_mean_reward = -np.inf
        self.metrics_path = run_dir / "evaluation_metrics.csv"

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        sync_envs_normalization(self.training_env, self.eval_env)
        self.eval_env.training = False
        self.eval_env.norm_reward = False

        successes = []

        def collect_success(locals_, globals_):
            if locals_.get("done", False):
                info = locals_.get("info", {})
                successes.append(float(np.asarray(
                    info.get("is_success", False)
                ).item()))

        rewards, lengths = evaluate_policy(
            self.model,
            self.eval_env,
            n_eval_episodes=self.n_eval_episodes,
            deterministic=True,
            callback=collect_success,
            return_episode_rewards=True,
            warn=False,
        )

        mean_reward = float(np.mean(rewards))
        std_reward = float(np.std(rewards))
        mean_length = float(np.mean(lengths))
        success_rate = float(np.mean(successes)) if successes else 0.0

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

        write_header = not self.metrics_path.exists()
        with self.metrics_path.open("a", newline="") as file:
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
            self.model.save(str(self.run_dir / "best_model"))
            self.training_env.save(
                str(self.run_dir / "best_vecnormalize.pkl")
            )
            record_evaluation_video(
                self.model,
                self.eval_env,
                self.run_dir / "videos" / "best_evaluation.mp4",
                self.max_episode_steps,
            )
            print("Saved new best model.")

        return True


def main():
    args = parse_args()

    run_dir = Path("artifacts/tqc") / args.run_name
    if args.resume_from is None:
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(
                f"Run directory already exists and is not empty: {run_dir}. "
                "Choose another --run-name or use --resume-from."
            )
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        if not run_dir.is_dir():
            raise FileNotFoundError(
                f"Run directory does not exist: {run_dir}. "
                "Use the original --run-name when resuming."
            )
        if not args.resume_from.is_file():
            raise FileNotFoundError(f"Resume model not found: {args.resume_from}")
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    tensorboard_dir = run_dir / "tensorboard"
    tensorboard_dir.mkdir(exist_ok=True)

    config_record = {
        "run_name": args.run_name,
        "seed": args.seed,
        "total_timesteps": args.total_timesteps,
        "environment": {
            "name": "MultiObjectPickAndPlaceEnv",
            "reward_type": args.reward_type,
            "max_episode_steps": args.max_episode_steps,
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
            "enabled": True,
            "strategy": args.her_strategy,
            "n_sampled_goal": args.her_sampled_goals,
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
        "exploration": "standard TQC policy; no FSM or custom action mixing",
    }
    config_path = run_dir / "config.json"
    if args.resume_from is None:
        with config_path.open("w") as file:
            json.dump(config_record, file, indent=2)
    elif not config_path.exists():
        raise FileNotFoundError(
            f"Missing original run configuration: {config_path}"
        )

    train_env = DummyVecEnv([
        make_env(
            run_dir / "training_monitor.csv",
            args.reward_type,
            args.max_episode_steps,
            render_mode="rgb_array",
        )
    ])
    train_env.seed(args.seed)
    if args.resume_from is None:
        train_env = VecNormalize(
            train_env,
            norm_obs=args.normalize_obs,
            norm_reward=args.normalize_reward,
            gamma=args.gamma,
        )
    else:
        resume_stem = args.resume_from.with_suffix("")
        resume_vecnormalize = resume_stem.parent / (
            resume_stem.name + "_vecnormalize.pkl"
        )
        resume_replay_buffer = resume_stem.parent / (
            resume_stem.name + "_replay_buffer.pkl"
        )
        for required_path in (resume_vecnormalize, resume_replay_buffer):
            if not required_path.is_file():
                raise FileNotFoundError(
                    f"Cannot resume: required state file is missing: {required_path}"
                )
        train_env = VecNormalize.load(str(resume_vecnormalize), train_env)
        train_env.training = True

    eval_env = DummyVecEnv([
        make_env(
            run_dir / "evaluation_monitor.csv",
            args.reward_type,
            args.max_episode_steps,
            render_mode="rgb_array",
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

    replay_buffer_kwargs = {
        "n_sampled_goal": args.her_sampled_goals,
        "goal_selection_strategy": args.her_strategy,
    }
    policy_kwargs = {
        "net_arch": args.net_arch,
        "n_critics": args.n_critics,
    }

    if args.resume_from is None:
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
            replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            tensorboard_log=str(tensorboard_dir),
            verbose=1,
            seed=args.seed,
            device=args.device,
        )
    else:
        model = TQC.load(
            str(args.resume_from),
            env=train_env,
            device=args.device,
            tensorboard_log=str(tensorboard_dir),
        )
        model.load_replay_buffer(str(resume_replay_buffer))
        print(f"Resumed model from: {args.resume_from}")
        print(f"Restored HER replay buffer: {resume_replay_buffer}")
        print(f"Restored VecNormalize state: {resume_vecnormalize}")

    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.checkpoint_freq, 1),
        save_path=str(checkpoint_dir),
        name_prefix="tqc_default",
        save_vecnormalize=True,
        save_replay_buffer=True,
        verbose=1,
    )
    eval_callback = SuccessRateEvalCallback(
        eval_env=eval_env,
        run_dir=run_dir,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.eval_episodes,
        max_episode_steps=args.max_episode_steps,
    )

    print("\nStandard TQC + HER: multi-object environment")
    print("=" * 60)
    print(f"Run directory: {run_dir}")
    print(f"Reward type: {args.reward_type}")
    print(f"Episode limit: {args.max_episode_steps}")
    print(f"Target cumulative timesteps: {args.total_timesteps}")
    print(f"Checkpoint frequency: {args.checkpoint_freq}")
    if args.resume_from is not None:
        print(f"Resume checkpoint: {args.resume_from}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning starts: {args.learning_starts}")
    print("FSM / custom exploration: disabled")
    print("=" * 60)

    def save_training_state(stem: str) -> None:
        """Save all state required for a full TQC + HER continuation."""
        model_path = run_dir / stem
        replay_buffer_path = run_dir / f"{stem}_replay_buffer.pkl"
        vecnormalize_path = run_dir / f"{stem}_vecnormalize.pkl"

        model.save(str(model_path))
        model.save_replay_buffer(str(replay_buffer_path))
        train_env.save(str(vecnormalize_path))

        print(f"Saved model: {model_path}.zip")
        print(f"Saved HER replay buffer: {replay_buffer_path}")
        print(f"Saved VecNormalize state: {vecnormalize_path}")

    try:
        remaining_timesteps = args.total_timesteps - model.num_timesteps
        if remaining_timesteps <= 0:
            raise ValueError(
                f"The loaded model already has {model.num_timesteps:,} timesteps, "
                f"which meets/exceeds the requested cumulative target "
                f"({args.total_timesteps:,}). Choose a larger --total-timesteps."
            )
        print(
            f"Starting at timestep {model.num_timesteps:,}; "
            f"training {remaining_timesteps:,} additional steps."
        )
        model.learn(
            total_timesteps=remaining_timesteps,
            reset_num_timesteps=(args.resume_from is None),
            callback=CallbackList([
                checkpoint_callback,
                eval_callback,
            ]),
            log_interval=10,
            tb_log_name="TQC_default",
        )

        save_training_state("final_model")
        # Keep the legacy filename for inference/evaluation scripts that expect it.
        train_env.save(str(run_dir / "vecnormalize.pkl"))
        record_evaluation_video(
            model,
            eval_env,
            run_dir / "videos" / "last_evaluation.mp4",
            args.max_episode_steps,
        )

        print("\nTraining completed.")
        print(f"Final model: {run_dir / 'final_model.zip'}")
    except KeyboardInterrupt:
        # Save a complete resumable snapshot even if Ctrl-C arrives between
        # periodic checkpoints.
        interrupted_stem = f"interrupted_{model.num_timesteps}_steps"
        print(
            f"\nCtrl-C received at timestep {model.num_timesteps:,}. "
            "Saving an interruption checkpoint..."
        )
        save_training_state(interrupted_stem)
        print(
            "\nTraining interrupted cleanly. Resume with --resume-from "
            f"{run_dir / (interrupted_stem + '.zip')}"
        )
    finally:
        train_env.close()
        eval_env.close()



if __name__ == "__main__":
    main()
