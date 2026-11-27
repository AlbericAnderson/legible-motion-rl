from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import panda_gym  # noqa: F401 - registers PandaGym environments

from sb3_contrib import TQC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, sync_envs_normalization
from stable_baselines3.her import HerReplayBuffer


ENV_IDS = {
    "dense": "PandaPickAndPlaceDense-v3",
    "sparse": "PandaPickAndPlace-v3",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Standard TQC + HER baseline on PandaGym PickAndPlace."
    )
    parser.add_argument("--reward-type", choices=("dense", "sparse"), required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    return parser.parse_args()


def make_env(env_id: str, monitor_file: Path, render_mode: str | None = None):
    def _init():
        env = gym.make(env_id, render_mode=render_mode)
        return Monitor(env, filename=str(monitor_file), info_keywords=("is_success",))
    return _init


class BestLastVideoEvalCallback(BaseCallback):
    """Evaluate deterministic policy and save last/best evaluation videos."""

    def __init__(
        self,
        eval_env: VecNormalize,
        run_dir: Path,
        eval_freq: int,
        n_eval_episodes: int,
        seed: int,
        video_fps: int = 25,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.run_dir = run_dir
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.seed = seed
        self.video_fps = video_fps
        self.best_success_rate = -1.0
        self.best_mean_reward = -np.inf
        self.metrics_path = run_dir / "evaluation_metrics.csv"
        self.last_video_path = run_dir / "last_evaluation.mp4"
        self.best_video_path = run_dir / "best_evaluation.mp4"

    def _normalize_obs(self, obs):
        if isinstance(obs, dict):
            batched = {key: np.asarray(value)[None, ...] for key, value in obs.items()}
        else:
            batched = np.asarray(obs)[None, ...]
        return self.eval_env.normalize_obs(batched)

    def _episode(self, episode_seed: int, record_video: bool = False):
        # The VecNormalize wrapper owns a DummyVecEnv containing a Monitor-wrapped
        # Gymnasium environment. Resetting the raw env lets us set an exact seed.
        raw_env = self.eval_env.venv.envs[0]
        obs, _ = raw_env.reset(seed=int(episode_seed))
        normalized_obs = self._normalize_obs(obs)

        frames = []
        if record_video:
            frame = raw_env.render()
            if frame is not None:
                frames.append(np.asarray(frame, dtype=np.uint8))

        total_reward = 0.0
        length = 0
        terminated = truncated = False
        info = {}

        while not (terminated or truncated):
            action, _ = self.model.predict(normalized_obs, deterministic=True)
            action = np.asarray(action)[0]
            obs, reward, terminated, truncated, info = raw_env.step(action)
            total_reward += float(reward)
            length += 1

            if record_video:
                frame = raw_env.render()
                if frame is not None:
                    frames.append(np.asarray(frame, dtype=np.uint8))

            if not (terminated or truncated):
                normalized_obs = self._normalize_obs(obs)

        success = float(np.asarray(info.get("is_success", False)).item())
        return total_reward, length, success, frames

    @staticmethod
    def _write_video(frames: list[np.ndarray], path: Path, fps: int):
        if not frames:
            raise RuntimeError(f"No frames were captured for {path}.")
        imageio.mimsave(path, frames, fps=fps, macro_block_size=1)

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        sync_envs_normalization(self.training_env, self.eval_env)
        self.eval_env.training = False
        self.eval_env.norm_reward = False

        rewards, lengths, successes = [], [], []
        best_episode_reward = -np.inf
        best_episode_seed = None
        seed_base = self.seed + int(self.num_timesteps)

        for i in range(self.n_eval_episodes):
            episode_seed = seed_base + i
            reward, length, success, _ = self._episode(episode_seed, record_video=False)
            rewards.append(reward)
            lengths.append(length)
            successes.append(success)
            if reward > best_episode_reward:
                best_episode_reward = reward
                best_episode_seed = episode_seed
            print(
                f"Eval {i + 1:02d}/{self.n_eval_episodes}: "
                f"reward={reward:.3f}, length={length}, success={bool(success)}"
            )

        mean_reward = float(np.mean(rewards))
        std_reward = float(np.std(rewards))
        mean_length = float(np.mean(lengths))
        success_rate = float(np.mean(successes))

        self.logger.record("eval/mean_reward", mean_reward)
        self.logger.record("eval/std_reward", std_reward)
        self.logger.record("eval/mean_episode_length", mean_length)
        self.logger.record("eval/success_rate", success_rate)

        print(
            f"\nEvaluation at {self.num_timesteps} steps | "
            f"reward={mean_reward:.3f} +/- {std_reward:.3f} | "
            f"length={mean_length:.1f} | success={success_rate:.1%}"
        )

        write_header = not self.metrics_path.exists()
        with self.metrics_path.open("a", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "timesteps", "mean_reward", "std_reward",
                    "mean_episode_length", "success_rate", "best_episode_reward",
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
                "best_episode_reward": best_episode_reward,
            })

        if best_episode_seed is None:
            raise RuntimeError("Evaluation did not select an episode.")

        replay_reward, replay_length, replay_success, frames = self._episode(
            best_episode_seed, record_video=True
        )
        if not np.isclose(replay_reward, best_episode_reward, rtol=1e-5, atol=1e-5):
            print(
                "Warning: replay reward differs from measured reward: "
                f"{replay_reward:.6f} vs {best_episode_reward:.6f}"
            )
        self._write_video(frames, self.last_video_path, self.video_fps)

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
            self.training_env.save(str(self.run_dir / "best_vecnormalize.pkl"))
            self._write_video(frames, self.best_video_path, self.video_fps)
            print("Saved new best model, VecNormalize statistics, and best video.")

        print(
            f"Saved last evaluation video: {self.last_video_path} "
            f"({replay_length} frames, success={bool(replay_success)})"
        )
        return True


def main():
    args = parse_args()
    if args.total_timesteps <= 0 or args.eval_freq <= 0 or args.eval_episodes <= 0:
        raise ValueError("Timesteps, evaluation frequency, and episode count must be positive.")

    env_id = ENV_IDS[args.reward_type]
    run_dir = Path("artifacts/tqc") / args.run_name
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Run directory already exists and is not empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = run_dir / "tensorboard"
    tb_dir.mkdir(exist_ok=True)

    config = {
        "algorithm": "TQC",
        "env_id": env_id,
        "reward_type": args.reward_type,
        "seed": args.seed,
        "total_timesteps": args.total_timesteps,
        "learning_rate": 1e-3,
        "buffer_size": 1_000_000,
        "learning_starts": 100,
        "batch_size": 2048,
        "tau": 0.05,
        "gamma": 0.95,
        "ent_coef": "auto",
        "policy": "MultiInputPolicy",
        "policy_kwargs": {"net_arch": [512, 512, 512], "n_critics": 2},
        "replay_buffer": "HerReplayBuffer",
        "replay_buffer_kwargs": {
            "n_sampled_goal": 4,
            "goal_selection_strategy": "future",
        },
        "normalize_obs": True,
        "normalize_reward": False,
        "norm_obs_keys": ["observation", "achieved_goal", "desired_goal"],
        "force_randomization": False,
        "residual_history": False,
        "evaluation": {
            "frequency": args.eval_freq,
            "episodes": args.eval_episodes,
            "deterministic": True,
            "videos": ["best_evaluation.mp4", "last_evaluation.mp4"],
        },
        "device": args.device,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    train_env = DummyVecEnv([
        make_env(
            env_id,
            run_dir / "training_monitor.csv",
            render_mode="rgb_array",
        )
    ])
        
    train_env.seed(args.seed)
    train_env = VecNormalize(
        train_env,
        norm_obs=True,
        norm_reward=False,
        gamma=0.95,
        norm_obs_keys=["observation", "achieved_goal", "desired_goal"],
    )

    eval_env = DummyVecEnv([
        make_env(
            env_id,
            run_dir / "evaluation_monitor.csv",
            render_mode="rgb_array",
        )
    ])
    eval_env.seed(args.seed + 10_000)
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,
        gamma=0.95,
        norm_obs_keys=["observation", "achieved_goal", "desired_goal"],
    )
    eval_env.training = False
    eval_env.norm_reward = False

    model = TQC(
        policy="MultiInputPolicy",
        env=train_env,
        learning_rate=1e-3,
        buffer_size=1_000_000,
        learning_starts=100,
        batch_size=2048,
        tau=0.05,
        gamma=0.95,
        ent_coef="auto",
        train_freq=1,
        gradient_steps=1,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs={
            "n_sampled_goal": 4,
            "goal_selection_strategy": "future",
        },
        policy_kwargs={"net_arch": [512, 512, 512], "n_critics": 2},
        tensorboard_log=str(tb_dir),
        verbose=1,
        seed=args.seed,
        device=args.device,
    )

    callback = BestLastVideoEvalCallback(
        eval_env=eval_env,
        run_dir=run_dir,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.eval_episodes,
        seed=args.seed + 10_000,
        video_fps=25,
        verbose=1,
    )

    print("=" * 64)
    print("Standard PandaGym TQC + HER baseline")
    print(f"Environment: {env_id}")
    print(f"Reward: {args.reward_type}")
    print(f"Timesteps: {args.total_timesteps}")
    print(f"Run directory: {run_dir}")
    print("FSM / custom actions / residual history: disabled")
    print("=" * 64)

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callback,
            log_interval=10,
            tb_log_name="TQC",
        )
        model.save(str(run_dir / "last_model"))
        train_env.save(str(run_dir / "last_vecnormalize.pkl"))
        print("Training completed.")
        print(f"Last model: {run_dir / 'last_model.zip'}")
        print(f"Best video: {run_dir / 'best_evaluation.mp4'}")
        print(f"Last video: {run_dir / 'last_evaluation.mp4'}")
    finally:
        train_env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
