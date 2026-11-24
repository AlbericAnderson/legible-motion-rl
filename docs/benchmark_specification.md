# Benchmark Specification

## 1. Benchmark description

A simulated multi-object, multi-destination pick-and-place task using a Franka Panda robot in PandaGym/PyBullet.

At the beginning of each episode:
- Multiple objects and destinations are placed in the workspace.
- One object–destination pair is randomly selected as the task goal.
- The robot receives the task goal and must physically pick the selected object and place it at the assigned destination.

The benchmark is used to study whether hierarchical observer feedback improves the legibility of robot motion.

The local-intention variable has five possible values:

`approach`, `grasp`, `lift`, `transport`, `place`.

Each episode has exactly one true goal:

\[
G=(i^\star,j^\star)
\]

where \(i^\star\) is the selected object and \(j^\star\) its assigned destination.

## 2. Current implementation specification

- Simulator: PandaGym 3.0.7 / PyBullet.
- Robot: Franka Panda.
- Control: Cartesian end-effector displacement and gripper control.
- Action space: 4 dimensions, bounded by [-1, 1].
- Physics timestep: 0.002 s, with 20 substeps.
- Control frequency: 25 Hz.
- Contact detection: Available through the underlying PyBullet client.
- Development baseline: `PandaPickAndPlace-v3`.

The existing PandaGym task is only a starting point. Its single-object observation and distance-based success criterion are insufficient for the final benchmark.

## 3. Information still to be specified

The following details are not yet finalized. They should be defined in this order:

1. Scene configuration: Number of objects and destinations, object geometry, dimensions, mass, and destination representation.
2. Randomization: Object and destination placement ranges, separation constraints, and goal-pair sampling.
3. Observation and action interfaces: Exact policy observations, observer-visible information, and final action representation.
4. Physical task mechanics: Grasp detection, lift validation, object transport, placement, and stability criteria.
5. Success and termination: Task success conditions, failure conditions, recovery behavior, and episode horizon.
6. Local-intention oracle: Deterministic predicates and priority rules for the five intention labels, including failed grasps and dropped objects.
7. Task reward: Reward components and termination-related penalties.
8. Validation criteria: Tests for environment behavior, randomization, physical manipulation, and oracle correctness.

