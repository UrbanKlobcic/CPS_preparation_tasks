# Drone Racing RL Project

## Task 1: Modeling of Environment Dynamics

## Task 2: Visualization of Rollouts with Rerun

For Task 2, we implemented a visualization tool using the Rerun logging framework to inspect drone racing rollouts in both 3D space and time-synchronized plots.

The core of the implementation is the function `visualize_state_action_sequence(sequence, gates, recording_path, app_id)`, which takes a rollout as a list of state–action pairs `(x, u)` and exports a `.rrd` recording. Each state `x` follows the provided 21-dimensional layout (position, velocity, acceleration, orientation quaternion, angular velocity, previous action, battery voltage), and each action `u` is a 4-dimensional normalized control command.

### Visualization design

The visualization is structured into static and dynamic components:

**Static scene elements (logged once):**
- A world coordinate frame using a right-handed, Z-up convention.
- Racing gate geometry constructed from the provided gate centers and orientations. Each gate is visualized as an outer and inner rectangular loop in 3D space.
- The full drone trajectory, visualized as a polyline connecting all position samples from the rollout.

**Dynamic elements (logged per timestep):**
- The drone pose over time, including motor positions, X-shaped arms, and a forward camera ray, derived from the drone’s position and orientation.
- Time-synchronized scalar plots for key quantities:
  - translational speed,
  - angular speed,
  - individual action components (roll, pitch, yaw, thrust),
  - Euler angles (roll, pitch, yaw).

A discrete time axis (`step`) is used to synchronize all spatial and scalar data, allowing smooth playback and frame-by-frame inspection in the Rerun viewer. An additional continuous time axis (`sim_time`) is logged using the simulator timestep (`dt = 0.01 s`).

### Usage

A small demo script is included via a `__main__` block. It loads an example rollout stored in a `.npy` file, converts it into the required `(x, u)` sequence format, and writes the visualization to `rollout.rrd`. The resulting file can be opened in the Rerun viewer to analyze drone motion, control behavior, and stability over time.

### Purpose

This visualization tool was used extensively to debug and analyze policies in Task 3. In particular, it helped identify issues such as unstable control, incorrect orientation handling, and inefficient gate approaches by correlating 3D trajectories with action and kinematic time-series.


## Task 3: Reinforcement Learning with PPO

### 3.1 Observation Design

An observation is a 31-dimensional vector expressed in the current gate's coordinate frame. It includes:

*   relative position (3D) and velocity (3D) in the current gate frame, scaled by 0.1 to keep magnitudes similar,
*   orientation (3D) relative to the current gate in Euler angles (roll, pitch, yaw) and angular rates (3D),
*   action-history buffer (3 * 4D) with the last 3 control commands, allows policy to infer the current state regardless of delays,
*   battery level (1D), allows policy to account for reduced thrust,
*   relative position of the next gate (3D) in the current gate frame,
*   next gate normal (3D) in the current gate frame. This was added to solve the blind cornering issue explained in section 3.4.

To improve robustness, we add noise to position, velocity, orientation, and angular rates.


### 3.2 Reward Design

We use the following for the reward shaping strategy:

*   **Progress Reward:** 1.0 per meter decrease in distance to the current gate between steps.
*   **Gate Bonus**: a large bonus (10.0) when the drone passes through a gate (distance below gate radius and/or
plane crossing).
*   **Speed Reward:** equal to the norm of the velocity vector scaled by 0.01.
*   **Survival Reward/Penalty:** a constant, small (0.001) penalty per step. Discourages the agent from hovering / looping in the same area. Initially configured as reward, but not necessary since crash penalty is enough to encourage survival.
*   **Altitude Penalty:** equal to the altitude difference between the drone and the current gate scaled by 0.01.
*   **Crash Penalty:** a large (10.0) penalty when the drone exceeds track bounds in any direction.
*   **Missed Gate Penalty:** a medium (2.0) penalty when the drone crosses the gate plane outside the gate radius.
*   **Control Smoothness Penalty:** a small per-step reward proportional to the norm of the control command. Disabled for initial training, could be turned on if movement is not smooth.
*   **Timeout Penalty:** a penalty when the drone exceeds the maximum episode length. Disabled for initial training.


### 3.3 Network Design

We use a simple actor-critic architecture with:

*   **Structure:** Both networks consist of 2 hidden layers with 512 units each.
*   **Layer Normalization:** We apply `LayerNorm` after every hidden layer since our observation space contains mixed magnitudes (e.g., battery voltage ~23 vs. Euler angles ~$\pi$).


### 3.4 Training Configuration

We trained using the `purejaxrl` PPO implementation with the following hyperparameters:

*   **`LR: 5e-4` with Linear Annealing:** We start with a relatively aggressive learning rate to quickly learn basic stabilization and navigation features. Annealing to 0 ensures convergence in the fine-tuning phase.
*   **`NUM_ENVS: 32`**: Large enough to approximate the true policy gradient, not too large for reasonable training on CPU.
*   **`NUM_STEPS: 1024`**: The rollout length (approx. 10 seconds). Long enough for a full lap, not too long for training on CPU.
*   **`TOTAL_TIMESTEPS: 1e8` (100M)**: Keeps training time reasonable, discussed in more detail in section 3.5.
*   **`GAMMA: 0.99`**: Discount factor. At 100Hz, $\gamma=0.99$ corresponds to a half-life of $\approx 1$ second.
*   **`GAE_LAMBDA: 0.95`**: Controls the bias-variance trade-off in advantage estimation. Higher values rely too heavily on future returns (noisy at the start of training), random crashes destabilize training. Lower values rely too heavily on the Critic's estimate, which is inaccurate early in training. 
*   **`CLIP_EPS: 0.2`**: Prevents new policy from diverging more than 20% from old policy in a single update. Larger values lead to inconsistency (and defeat the point of PPO), smaller values slow down training.
*   **`ENT_COEF: 0.02`**: Entropy value of 0.02 keeps the policy stochastic enough to explore alternative trajectories (e.g., taking a wider turn). Larger values lead to inconsistency (random crashes), smaller values lead to local optima, like slowing down a lot before a tight turn instead of taking a wider turn.
*   **`VF_COEF: 0.5`**: Scales the Value Function loss relative to the Policy loss. This allows us to control whether to prioritize learning for the Actor or the Critic. Balanced training (0.5) works well.
*   **`MAX_GRAD_NORM: 0.5`**: Prevents spikes in gradients (e.g. from crashes) from destroying the policy.


### 3.5 Discussion

