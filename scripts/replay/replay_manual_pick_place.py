import csv
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from legible_motion.envs import MultiObjectPickAndPlaceEnv


OUTPUT_DIR = Path("artifacts/replays/manual_pick_place")
VIDEO_PATH = OUTPUT_DIR / "manual_pick_place.mp4"
CSV_PATH = OUTPUT_DIR / "manual_pick_place.csv"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"

FPS = 25


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    env = MultiObjectPickAndPlaceEnv(
        render_mode="rgb_array",
        renderer="Tiny",
    )

    frames = []
    records = []
    step_count = 0

    try:
        env.reset(seed=42)

        physics = env.sim.physics_client
        task = env.task

        # -------------------------------------------------
        # Selected object and destination
        # -------------------------------------------------

        object_name = task.object_names[task.goal_object_idx]
        object_id = env.sim._bodies_idx[object_name]

        destination_name = f"target_{task.goal_destination_idx}"
        destination_id = env.sim._bodies_idx[destination_name]

        panda_id = env.sim._bodies_idx["panda"]

        goal = np.asarray(task.get_goal())

        object_half_size = task.config.object_size / 2
        destination_half_extents = np.asarray(
            task.config.destination_half_extents
        )

        print("Selected object:", object_name)
        print("Selected destination:", destination_name)
        print("Initial object:", task.get_achieved_goal())
        print("Destination:", goal)

        # -------------------------------------------------
        # State and recording helpers
        # -------------------------------------------------

        def get_object_position():
            position, _ = physics.getBasePositionAndOrientation(object_id)
            return np.asarray(position)

        def capture_frame():
            frame = env.render()

            if frame is None:
                raise RuntimeError(
                    "env.render() returned None. "
                    "Check the environment rendering configuration."
                )

            frames.append(np.asarray(frame).copy())

        def record_step(action, phase):
            nonlocal step_count

            action = np.asarray(action, dtype=float)

            observation, reward, terminated, truncated, info = env.step(
                action
            )

            step_count += 1

            object_position = get_object_position()
            ee_position = np.asarray(env.robot.get_ee_position())

            linear_velocity, angular_velocity = physics.getBaseVelocity(
                object_id
            )

            contacts = physics.getContactPoints(
                bodyA=panda_id,
                bodyB=object_id,
            )

            distance = float(
                np.linalg.norm(object_position - goal)
            )

            records.append({
                "step": step_count,
                "phase": phase,

                "action_x": action[0],
                "action_y": action[1],
                "action_z": action[2],
                "action_gripper": action[3],

                "ee_x": ee_position[0],
                "ee_y": ee_position[1],
                "ee_z": ee_position[2],

                "object_x": object_position[0],
                "object_y": object_position[1],
                "object_z": object_position[2],

                "target_x": goal[0],
                "target_y": goal[1],
                "target_z": goal[2],

                "ee_object_offset_x": (
                    ee_position[0] - object_position[0]
                ),
                "ee_object_offset_y": (
                    ee_position[1] - object_position[1]
                ),
                "ee_object_offset_z": (
                    ee_position[2] - object_position[2]
                ),

                "object_target_distance": distance,

                "object_vx": linear_velocity[0],
                "object_vy": linear_velocity[1],
                "object_vz": linear_velocity[2],

                "object_wx": angular_velocity[0],
                "object_wy": angular_velocity[1],
                "object_wz": angular_velocity[2],

                "finger_width": env.robot.get_fingers_width(),
                "robot_object_contacts": len(contacts),

                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "is_success": bool(info.get("is_success", False)),
            })

            capture_frame()

            return observation, reward, terminated, truncated, info

        def diagnostics(label):
            object_position = get_object_position()
            ee_position = np.asarray(env.robot.get_ee_position())

            linear_velocity, angular_velocity = physics.getBaseVelocity(
                object_id
            )

            contacts = physics.getContactPoints(
                bodyA=panda_id,
                bodyB=object_id,
            )

            print(f"\n--- {label} ---")
            print("Object:", np.round(object_position, 4))
            print("EE:", np.round(ee_position, 4))
            print(
                "EE-object XY offset:",
                np.round(ee_position[:2] - object_position[:2], 4),
            )
            print(
                "Object-target XY error:",
                np.round(goal[:2] - object_position[:2], 4),
            )
            print(
                "Object-target distance:",
                round(np.linalg.norm(object_position - goal), 4),
            )
            print("Linear velocity:", np.round(linear_velocity, 4))
            print("Angular velocity:", np.round(angular_velocity, 4))
            print("Robot-object contacts:", len(contacts))
            print("Gripper width:", round(env.robot.get_fingers_width(), 4))

            return object_position

        # Capture initial state.
        capture_frame()

        # -------------------------------------------------
        # Open gripper
        # -------------------------------------------------

        for _ in range(4):
            record_step([0, 0, 0, 1], "open_gripper")

        # -------------------------------------------------
        # Align with object
        # -------------------------------------------------

        object_position = get_object_position()
        target_xy = object_position[:2].copy()

        for _ in range(15):
            ee = np.asarray(env.robot.get_ee_position())
            error = target_xy - ee[:2]

            if np.linalg.norm(error) < 0.002:
                break

            action_xy = np.clip(error / 0.05, -1, 1)

            record_step([
                action_xy[0],
                action_xy[1],
                0,
                0,
            ], "approach")

        # -------------------------------------------------
        # Descend
        # -------------------------------------------------

        for _ in range(40):
            ee = np.asarray(env.robot.get_ee_position())
            error_z = 0.025 - ee[2]

            if abs(error_z) < 0.003:
                break

            action_z = np.clip(error_z / 0.05, -0.1, 0.1)

            record_step(
                [0, 0, action_z, 0],
                "descend",
            )

        # -------------------------------------------------
        # Grasp
        # -------------------------------------------------

        for _ in range(8):
            record_step(
                [0, 0, 0, -0.25],
                "grasp",
            )

        diagnostics("After grasp")

        # -------------------------------------------------
        # Lift
        # -------------------------------------------------

        for _ in range(30):
            ee = np.asarray(env.robot.get_ee_position())
            error_z = 0.12 - ee[2]

            if abs(error_z) < 0.003:
                break

            action_z = np.clip(error_z / 0.05, -0.1, 0.1)

            record_step(
                [0, 0, action_z, -0.25],
                "lift",
            )

        diagnostics("After lift")

        # -------------------------------------------------
        # Transport to selected destination
        # -------------------------------------------------

        object_position = get_object_position()
        ee = np.asarray(env.robot.get_ee_position())

        grasp_offset = ee[:2] - object_position[:2]
        target_ee_xy = goal[:2] + grasp_offset

        print("\nTransport target EE XY:", np.round(target_ee_xy, 4))

        for _ in range(100):
            ee = np.asarray(env.robot.get_ee_position())
            error_xy = target_ee_xy - ee[:2]

            if np.linalg.norm(error_xy) < 0.003:
                break

            action_xy = np.clip(error_xy / 0.05, -0.1, 0.1)

            record_step([
                action_xy[0],
                action_xy[1],
                0,
                -0.25,
            ], "transport")

        diagnostics("After transport")

        # -------------------------------------------------
        # Lower while holding the object
        # -------------------------------------------------

        target_z = 0.025

        for _ in range(40):
            ee = np.asarray(env.robot.get_ee_position())
            error_z = target_z - ee[2]

            if abs(error_z) < 0.003:
                break

            action_z = np.clip(error_z / 0.05, -0.1, 0.1)

            record_step(
                [0, 0, action_z, -0.25],
                "lower",
            )

        diagnostics("Immediately before release")

        # -------------------------------------------------
        # Release
        # -------------------------------------------------

        for _ in range(8):
            record_step(
                [0, 0, 0, 0.25],
                "release",
            )

        diagnostics("Immediately after release")

        # -------------------------------------------------
        # Settle
        # -------------------------------------------------

        for _ in range(20):
            record_step(
                [0, 0, 0, 0],
                "settle",
            )

        final_object = diagnostics("After settling")

        # -------------------------------------------------
        # Placement validation
        # -------------------------------------------------

        xy_error = final_object[:2] - goal[:2]

        containment_limits = (
            destination_half_extents - object_half_size
        )

        contained = bool(
            np.all(np.abs(xy_error) <= containment_limits)
        )

        final_distance = float(
            np.linalg.norm(final_object - goal)
        )

        final_linear_velocity, final_angular_velocity = (
            physics.getBaseVelocity(object_id)
        )

        final_contacts = physics.getContactPoints(
            bodyA=panda_id,
            bodyB=object_id,
        )

        environment_success = bool(
            task.is_success(final_object, goal)
        )

        summary = {
            "seed": 42,
            "object": object_name,
            "destination": destination_name,
            "steps": step_count,
            "fps": FPS,

            "initial_object_position": (
                task.object_positions[task.goal_object_idx].tolist()
            ),
            "target_position": goal.tolist(),
            "final_object_position": final_object.tolist(),

            "final_xy_error": xy_error.tolist(),
            "final_distance": final_distance,

            "containment_limits": containment_limits.tolist(),
            "object_fully_inside_destination": contained,

            "environment_is_success": environment_success,
            "final_robot_object_contacts": len(final_contacts),

            "final_linear_velocity": (
                np.asarray(final_linear_velocity).tolist()
            ),
            "final_angular_velocity": (
                np.asarray(final_angular_velocity).tolist()
            ),

            "gripper_width": float(env.robot.get_fingers_width()),
        }

        print("\n========== PLACEMENT VALIDATION ==========")
        print("Final object:", np.round(final_object, 4))
        print("Destination:", np.round(goal, 4))
        print("XY error:", np.round(xy_error, 4))
        print("Final distance:", round(final_distance, 4), "m")
        print("Object fully inside destination:", contained)
        print("Environment is_success:", environment_success)

        # -------------------------------------------------
        # Save video
        # -------------------------------------------------

        imageio.mimsave(
            VIDEO_PATH,
            frames,
            fps=FPS,
            macro_block_size=1,
        )

        # -------------------------------------------------
        # Save per-step CSV
        # -------------------------------------------------

        if records:
            with CSV_PATH.open("w", newline="") as csv_file:
                writer = csv.DictWriter(
                    csv_file,
                    fieldnames=records[0].keys(),
                )
                writer.writeheader()
                writer.writerows(records)

        # -------------------------------------------------
        # Save summary
        # -------------------------------------------------

        SUMMARY_PATH.write_text(
            json.dumps(summary, indent=4)
        )

        print("\n========== SAVED ARTIFACTS ==========")
        print("Video:", VIDEO_PATH)
        print("CSV:", CSV_PATH)
        print("Summary:", SUMMARY_PATH)
        print("Frames:", len(frames))
        print("Recorded steps:", len(records))

    finally:
        env.close()


if __name__ == "__main__":
    main()