from __future__ import annotations

import argparse
import csv
from pathlib import Path

import gymnasium as gym
import numpy as np
from sb3_contrib import TQC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from legible_motion.envs.multi_object import (
    MultiObjectConfig,
    MultiObjectPickAndPlaceEnv,
)


DEFAULT_PANDAGYM_ENV_ID = "PandaPickAndPlace-v3"


class TerminalFrameCapture(gym.Wrapper):
    """Capture the terminal image before a VecEnv can automatically reset."""

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            info = dict(info)
            frame = self.env.render()
            if frame is not None:
                info["terminal_frame"] = np.asarray(frame).copy()
        return observation, reward, terminated, truncated, info


def make_environment(args):
    if args.environment == "multi_object":
        config = MultiObjectConfig(
            reward_type=args.reward_type,
            max_episode_steps=args.max_episode_steps,
        )
        base_env = MultiObjectPickAndPlaceEnv(
            config=config,
            render_mode="rgb_array",
        )
    else:
        # Keep PandaGym's registered default task and time-limit settings.
        base_env = gym.make(
            args.pandagym_env_id,
            render_mode="rgb_array",
        )

    return Monitor(TerminalFrameCapture(base_env))


def space_signature(space):
    """Return a concise, recursive description for compatibility checks."""
    if isinstance(space, gym.spaces.Dict):
        return {key: space_signature(subspace) for key, subspace in space.spaces.items()}
    if isinstance(space, gym.spaces.Tuple):
        return tuple(space_signature(subspace) for subspace in space.spaces)

    signature = {
        "type": type(space).__name__,
        "shape": getattr(space, "shape", None),
        "dtype": str(getattr(space, "dtype", None)),
    }
    if isinstance(space, gym.spaces.Box):
        signature["low"] = np.asarray(space.low)
        signature["high"] = np.asarray(space.high)
    return signature


def compare_spaces(model_space, env_space, label):
    """Raise a useful error if checkpoint and evaluation spaces differ."""
    model_sig = space_signature(model_space)
    env_sig = space_signature(env_space)

    def compare(left, right, path=""):
        # NumPy arrays
        if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
            if not isinstance(left, np.ndarray) or not isinstance(right, np.ndarray):
                return f"{path}: different types ({type(left)} != {type(right)})"

            if not np.array_equal(left, right):
                return f"{path}: array values differ"

            return None

        # Dictionaries
        if isinstance(left, dict) and isinstance(right, dict):
            if left.keys() != right.keys():
                return f"{path}: dictionary keys differ"

            for key in left:
                child_path = f"{path}.{key}" if path else str(key)

                problem = compare(left[key], right[key], child_path)

                if problem:
                    return problem

            return None

        # Lists and tuples
        if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
            if len(left) != len(right):
                return f"{path}: sequence lengths differ"

            for index, (l_item, r_item) in enumerate(zip(left, right)):
                problem = compare(l_item, r_item, f"{path}[{index}]")

                if problem:
                    return problem

            return None

        # Scalar values
        if left != right:
            return f"{path}: values differ ({left} != {right})"

        return None

    problem = compare(model_sig, env_sig)
    if problem:
        raise ValueError(problem)


def save_video(frames, output_path: Path, fps: int):
    if not frames:
        return
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise RuntimeError(
            "Video recording requires imageio. Install it with: pip install imageio"
        ) from exc
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_path, frames, fps=fps)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a TQC checkpoint in PandaGym or the multi-object environment."
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--vecnormalize", type=str, required=True)
    parser.add_argument(
        "--environment",
        choices=["multi_object", "panda_default"],
        required=True,
        help="Environment used to evaluate the checkpoint.",
    )
    parser.add_argument(
        "--pandagym-env-id",
        type=str,
        default=DEFAULT_PANDAGYM_ENV_ID,
        help="Registered PandaGym ID for --environment panda_default.",
    )
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--reward-type", choices=["dense", "sparse"], default="dense",
        help="Reward type for the custom multi-object environment.",
    )
    parser.add_argument(
        "--max-episode-steps", type=int, default=60,
        help="Horizon for the custom environment; PandaGym's registered horizon is unchanged.",
    )
    parser.add_argument(
        "--record-video",
        action="store_true",
        help="Record evaluation videos.",
    )
    parser.add_argument(
        "--video-mode",
        choices=["best", "all"],
        default="best",
        help=(
            "When recording videos, save only the best episode or every episode. "
            "Default: best."
        ),
    )
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Defaults to <model directory>/evaluation_<environment>/.",
    )
    args = parser.parse_args()

    if args.episodes < 1:
        parser.error("--episodes must be at least 1.")
    if args.max_episode_steps < 1:
        parser.error("--max-episode-steps must be at least 1.")
    if args.video_fps < 1:
        parser.error("--video-fps must be at least 1.")
    if args.video_mode == "all" and not args.record_video:
        parser.error("--video-mode all requires --record-video.")

    model_path = Path(args.model)
    vecnormalize_path = Path(args.vecnormalize)
    if not model_path.is_file():
        parser.error(f"Model file not found: {model_path}")
    if not vecnormalize_path.is_file():
        parser.error(f"VecNormalize file not found: {vecnormalize_path}")

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else model_path.parent / f"evaluation_{args.environment}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_env = DummyVecEnv([lambda: make_environment(args)])
    try:
        env = VecNormalize.load(str(vecnormalize_path), raw_env)
    except Exception:
        raw_env.close()
        raise

    env.training = False
    env.norm_reward = False
    env.seed(args.seed)

    # Load the model with the environment
    model = TQC.load(
        str(model_path),
        env=env,
        device=args.device,
    )

    try:
        compare_spaces(model.observation_space, env.observation_space, "Observation space")
        compare_spaces(model.action_space, env.action_space, "Action space")
    except ValueError as exc:
        env.close()
        raise SystemExit(
            "\nCheckpoint/environment incompatibility:\n"
            f"{exc}\n\n"
            "Use the VecNormalize statistics and environment matching the checkpoint's "
            "training setup. Matching dimensions alone do not guarantee matching "
            "observation semantics."
        ) from exc

    model.set_env(env)

    episode_rewards = []
    episode_lengths = []
    episode_successes = []

    # In "all" mode, write each episode immediately to avoid retaining
    # every video's frames in memory. In "best" mode, retain only the
    # current best episode.
    videos_dir = output_dir / "videos"
    best_video_frames = None
    best_video_episode = None
    best_video_score = None

    for episode in range(args.episodes):
        obs = env.reset()
        done = False
        total_reward = 0.0
        episode_length = 0
        frames = []

        if args.record_video:
            initial_frame = env.envs[0].render()
            if initial_frame is not None:
                frames.append(np.asarray(initial_frame).copy())

        final_info = {}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)
            total_reward += float(rewards[0])
            episode_length += 1
            done = bool(dones[0])
            final_info = infos[0]

            if args.record_video:
                if done:
                    # VecEnv has already reset the raw environment. Use the frame
                    # captured inside TerminalFrameCapture before that reset.
                    terminal_frame = final_info.get("terminal_frame")
                    if terminal_frame is not None:
                        frames.append(np.asarray(terminal_frame).copy())
                else:
                    frame = env.envs[0].render()
                    if frame is not None:
                        frames.append(np.asarray(frame).copy())

        success_value = final_info.get("is_success", False)
        success = bool(np.asarray(success_value).item())
        episode_rewards.append(total_reward)
        episode_lengths.append(episode_length)
        episode_successes.append(float(success))

        print(
            f"Episode {episode + 1:03d}/{args.episodes}: "
            f"reward={total_reward:.3f}, length={episode_length}, success={success}"
        )

        if args.record_video:
            if args.video_mode == "all":
                video_path = videos_dir / f"episode_{episode + 1:03d}.mp4"
                save_video(frames, video_path, args.video_fps)
                print(f"Episode video saved: {video_path}")
            else:
                # Prefer successful episodes; break ties using cumulative reward.
                video_score = (int(success), total_reward)

                if best_video_score is None or video_score > best_video_score:
                    best_video_score = video_score
                    best_video_frames = frames
                    best_video_episode = episode + 1

    if args.record_video and args.video_mode == "best" and best_video_frames:
        video_path = output_dir / "best_episode.mp4"
        save_video(best_video_frames, video_path, args.video_fps)
        print(
            f"Best episode video saved: {video_path} "
            f"(episode {best_video_episode}, "
            f"success={bool(best_video_score[0])}, "
            f"reward={best_video_score[1]:.3f})"
        )

    mean_reward = float(np.mean(episode_rewards))
    std_reward = float(np.std(episode_rewards))
    mean_length = float(np.mean(episode_lengths))
    success_rate = float(np.mean(episode_successes))

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Environment: {args.environment}")
    print(f"Episodes: {args.episodes}")
    print(f"Mean reward: {mean_reward:.3f} +/- {std_reward:.3f}")
    print(f"Mean episode length: {mean_length:.1f}")
    print(f"Success rate: {success_rate * 100:.1f}%")
    print("=" * 60)

    results_file = output_dir / "evaluation_results.csv"
    with results_file.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["episode", "reward", "length", "success"])
        for index in range(args.episodes):
            writer.writerow([
                index + 1,
                episode_rewards[index],
                episode_lengths[index],
                episode_successes[index],
            ])

    print(f"Results saved to: {results_file}")
    env.close()


if __name__ == "__main__":
    main()
