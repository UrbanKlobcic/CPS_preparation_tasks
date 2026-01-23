"""
DroneRaceEnv (TEMPLATE)

This environment has following functions:
- reset() and step() are pure functions (no side effects)
- suitable for vectorization (vmap) and JIT compilation
- designed to be used with a pure-JAX PPO implementation

Implement the TODOs, especially:
- reward shaping
- termination conditions
- (optional observation noise and action-history features)
- (optional domain randomization)

State/action conventions match the course handout:

State x (shape (21,)):
  [x, y, z,                                                 Position (m)
   vx, vy, vz,                                              Velocity (m/s)
   ax, ay, az,                                              Acceleration (m/s^2)
   qw, qx, qy, qz,                                          Quaternion orientation
   wx, wy, wz,                                              Angular velocity (rad/s)
   u_roll_prev, u_pitch_prev, u_yaw_prev, u_thrust_prev,    Previous action (applied)
   battery_V]                                               Battery voltage (V)

Action u (shape (4,)):
  [u_roll, u_pitch, u_yaw, u_thrust] in [-1, 1]
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from utils import *
from gymnax.environments import spaces


# ---------------------------------------------------------------------
# TODO: import learned dynamics model from Task 1
# ---------------------------------------------------------------------
# Your Task-1 model should have signature:
#   x_next = step(x, u, dt, params)
#
# Example (adjust to your project structure):
# from envs.models.dynamics import step as dynamics_step, DEFAULT_PARAMS, ModelParameters
#
class ModelParams(NamedTuple):
    # Physics constants
    m: float = 1.0
    g: float = 9.81
    # Action delay (first order lag)
    t_delay: float = 0.02 
    # Rate limits
    max_rate: jax.Array = jnp.array([618.0, 618.0, 120.0]) # deg/s
    # Thrust curve (approximate)
    # T = (u_th * c1 + c0) * V / V_ref... simplified for training:
    thrust_scale: float = 40.0  # Max thrust in Newtons (approx TWR=4)
    thrust_min: float = 0.0

def dynamics_step(x: jax.Array, u: jax.Array, dt: float, params) -> jax.Array:
    """
    Pure JAX implementation of Quadcopter dynamics.
    x: [pos(3), vel(3), acc(3), q(4), w(3), u_prev(4), V(1)]
    """
    # Constants
    m = 1.0
    g = 9.81
    max_rates = jnp.array([10.78, 10.78, 2.09]) # rad/s (approx 618 deg/s)
    
    # 1. Action Delay (Low pass filter)
    # alpha = dt / (dt + tau) or similar. Using simple lerp:
    alpha = 0.6 # Roughly corresponds to 0.01s lag at 100hz
    u_prev = x[16:20]
    u_eff = u_prev * (1 - alpha) + u * alpha
    
    # 2. Angular Dynamics (Acro Mode)
    # Map action [-1, 1] to target body rates
    w_target = u_eff[0:3] * max_rates
    
    # Simple P-controller or direct rate setting for body rates 
    # (The handout says rate commands map linearly, so we assume w = w_target for simplicity 
    #  or add a slight drag/inertia, but direct mapping is fine for PPO baseline)
    w_next = w_target 

    # 3. Orientation Update (Quaternion Integration)
    # q_new = q + 0.5 * q * w * dt
    q = x[9:13]
    # Create pure quaternion from w
    w_quat = jnp.array([0.0, w_next[0], w_next[1], w_next[2]])
    # dq = 0.5 * q * w
    dq = 0.5 * quat_mul(q, w_quat)
    q_next = quat_normalize(q + dq * dt)
    
    # 4. Linear Dynamics
    # Thrust model: simplified linear map from [-1, 1] to [0, Max]
    # u_thrust [-1, 1] -> [0, 1]
    throttle = (u_eff[3] + 1.0) / 2.0
    thrust_mag = throttle * 40.0 # Approx 40N max thrust
    
    # Rotate thrust vector (0,0,T) by quaternion to world frame
    thrust_local = jnp.array([0.0, 0.0, thrust_mag])
    thrust_world = quat_rotate(q_next, thrust_local) # Function from utils.py
    
    # Acceleration = F/m - g
    acc = thrust_world / m - jnp.array([0.0, 0.0, g])
    
    # Drag (Simple linear air resistance)
    vel = x[3:6]
    drag = -0.1 * vel
    acc = acc + drag
    
    vel_next = vel + acc * dt
    pos_next = x[0:3] + vel * dt + 0.5 * acc * dt**2
    
    # 5. Battery (Simple drain)
    voltage = x[20] - 0.001 # discharge
    
    # Pack State
    x_next = jnp.concatenate([
        pos_next,    # 0-2
        vel_next,    # 3-5
        acc,         # 6-8
        q_next,      # 9-12
        w_next,      # 13-15
        u_eff,       # 16-19 (Store effective input as prev)
        jnp.array([voltage]) # 20
    ])
    
    return x_next


SIM_HZ = 100
SIM_DT = 1.0 / SIM_HZ

START_POS = jnp.array([18.6, 2.0, 0.1], dtype=jnp.float32)

# gates: [id, x, y, z, qw, qx, qy, qz]
ENVIRONMENT = jnp.array(
    [
        [0, 12.500000, 2.000000, 1.350000, -0.707107, 0.000000, 0.000000, 0.707107],  # yaw = 270.00
        [1, 6.500000, 6.000000, 1.350000, -0.382684, 0.000000, 0.000000, 0.923879],  # yaw = 225.00
        [2, 5.500000, 14.000000, 1.350000, -0.258819, 0.000000, 0.000000, 0.965926],  # yaw = 210.00
        [3, 2.500000, 24.000000, 1.350000, 0.000000, 0.000000, 0.000000, 1.000000],  # yaw = 180.00
        [4, 7.500000, 30.000000, 1.350000, -0.642788, 0.000000, 0.000000, 0.766044],  # yaw = 260.00
        [8, 18.500000, 22.000000, 1.350000, -0.087155, 0.000000, 0.000000, 0.996195],  # yaw = 190.00
        [9, 20.500000, 14.000000, 1.350000, 0.087155, 0.000000, 0.000000, 0.996195],  # yaw = 170.00
        [10, 18.500000, 6.000000, 1.350000, 0.382684, 0.000000, 0.000000, 0.923879],  # yaw = 135.00
    ],
    dtype=jnp.float32,
)

MIN_POS = jnp.min(ENVIRONMENT[:, 1:4], axis=0)
MAX_POS = jnp.max(ENVIRONMENT[:, 1:4], axis=0)
BOUNDS = jnp.stack([MIN_POS - 5.0, MAX_POS + 5.0], axis=0)

NUM_GATES = ENVIRONMENT.shape[0]

BOUNDS_LOW = MIN_POS - jnp.array([5.0, 5.0, 2.0], dtype=jnp.float32)
BOUNDS_HIGH = MAX_POS + jnp.array([5.0, 5.0, 5.0], dtype=jnp.float32)

MIN_ALTITUDE = 0.15
MAX_SPEED = 15.0
BATTERY_MIN = 22.0
BATTERY_MAX = 24.0


class EnvParams(NamedTuple):
    gate_radius: float = 0.75
    max_episode_steps: int = SIM_HZ * 20

    # reward weights
    w_gate: float = 50.0
    w_progress: float = 1.0
    w_survival: float = 1.0

    # penalty weights
    w_control: float = 0.05
    w_altitude: float = 0.5
    w_speed: float = 0.0
    w_missed_gate: float = 5.0
    w_crash: float = 10.0
    w_timeout: float = 0.0


DEFAULT_PARAMS = EnvParams()


def compute_gate_forward(gate_quat: jax.Array) -> jax.Array:
    return quat_rotate(gate_quat, jnp.array([1.0, 0.0, 0.0], dtype=jnp.float32))


def get_gate_pose(gate_index: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Return (gate_pos, gate_quat) for the given index."""
    idx = gate_index.astype(jnp.int32)
    gate_row = ENVIRONMENT[idx]
    pos = gate_row[1:4]
    quat = gate_row[4:8]
    quat = quat / (jnp.linalg.norm(quat) + 1e-8)
    return pos, quat


def world_to_gate_position(pos_world: jax.Array, gate_pos: jax.Array, gate_quat: jax.Array) -> jax.Array:
    """Transform a world-frame position into the gate frame (gate at origin)."""
    q_inv = quat_conjugate(gate_quat)
    return quat_rotate(q_inv, pos_world - gate_pos)


def world_to_gate_vector(vec_world: jax.Array, gate_quat: jax.Array) -> jax.Array:
    """Transform a world-frame vector into the gate frame."""
    q_inv = quat_conjugate(gate_quat)
    return quat_rotate(q_inv, vec_world)


# Env state container
class EnvState(NamedTuple):
    x: jax.Array
    gate_index: jax.Array
    step_count: jax.Array
    prev_dist_to_gate: jax.Array
    gates_passed: jax.Array


class DroneRaceEnv:
    """
    Minimal racing environment skeleton.

    Required for the exercise:
      - observation in *current gate frame*
      - observation includes *relative position of the next gate*
      - reward encourages progressing through gates
      - termination on time limit and safety bounds

    Suggestions (optional):
      - include previous actions (x[16:20]) and/or action history buffer
      - add observation noise
      - add domain randomization sampled per episode
    """

    obs_size: int = 20
    action_size: int = 4

    @property
    def default_params(self) -> EnvParams:
        return DEFAULT_PARAMS


    def action_space(self, params=None):
        return spaces.Box(low=-1.0, high=1.0, shape=(4,))


    def observation_space(self, params=None):
        return spaces.Box(
            low=-jnp.inf * jnp.ones((self.obs_size,), dtype=jnp.float32),
            high=jnp.inf * jnp.ones((self.obs_size,), dtype=jnp.float32),
            shape=(self.obs_size,),
        )


    def reset(self, rng: jax.Array, params) -> tuple[jax.Array, EnvState]:
        """
        Reset environment and return (obs, state).

        TODO:
          1) Create an initial drone state x0, shape (21,):
             - position: START_POS
             - orientation: face gate 0 (use yaw_to_quat and a look-at direction)
             - battery voltage: e.g., 24.0
             - optionally set velocity to 0 and previous action to 0
          2) Set gate_index = 0, step_count = 0
          3) Compute observation in current gate frame via _obs_from_state()

        Optional (recommended):
          - randomize initial positions/orientation slightly (robustness)
          - do not always start at gate 0 but slightly after a random gate (strongly recommended)
          - sample domain randomization parameters and store them in EnvState
        """
        rng, rng_pos, rng_yaw, rng_gate = jax.random.split(rng, 4)

        start_gate_idx = jax.random.randint(rng_gate, (), 0, NUM_GATES)

        gate_pos, gate_quat = get_gate_pose(start_gate_idx)
        gate_forward = compute_gate_forward(gate_quat)

        offset_behind = -gate_forward * 3.0
        lateral_offset = jax.random.uniform(rng_pos, (3,), minval=-0.5, maxval=0.5)
        lateral_offset = lateral_offset.at[2].set(
            jax.random.uniform(rng_pos, (), minval=-0.3, maxval=0.3)
        )

        init_pos = gate_pos + offset_behind + lateral_offset
        init_pos = init_pos.at[2].set(jnp.clip(init_pos[2], 0.5, 2.5))

        target_dir = gate_pos - init_pos
        target_yaw = jnp.arctan2(target_dir[1], target_dir[0])
        yaw_noise = jax.random.uniform(rng_yaw, (), minval=-0.2, maxval=0.2)
        init_quat = yaw_to_quat(target_yaw + yaw_noise)

        x0 = jnp.zeros(21, dtype=jnp.float32)
        x0 = x0.at[0:3].set(init_pos)
        x0 = x0.at[3:6].set(jnp.zeros(3, dtype=jnp.float32))
        x0 = x0.at[6:9].set(jnp.zeros(3, dtype=jnp.float32))
        x0 = x0.at[9:13].set(init_quat)
        x0 = x0.at[13:16].set(jnp.zeros(3, dtype=jnp.float32))
        x0 = x0.at[16:20].set(jnp.array([0.0, 0.0, 0.0, -1.0], dtype=jnp.float32))
        x0 = x0.at[20].set(BATTERY_MAX)

        init_dist = jnp.linalg.norm(init_pos - gate_pos)

        state = EnvState(
            x=x0,
            gate_index=start_gate_idx.astype(jnp.float32),
            step_count=jnp.array(0, dtype=jnp.float32),
            prev_dist_to_gate=init_dist,
            gates_passed=jnp.array(0, dtype=jnp.float32),
        )

        obs = self._obs_from_state(state)
        return obs, state


    def step(self, rng: jax.Array, state: EnvState, action: jax.Array, params: EnvParams):
        """
        Step the environment and return (obs, next_state, reward, done, info).

        TODO:
          1) Clip/squash action to [-1, 1]
          2) Apply dynamics:
               x_next = dynamics_step(x, u, SIM_DT, params)
          3) Gate update:
               - compute distance to current gate in gate frame
               - passed_gate = dist < gate_radius
               - if passed_gate: gate_index <- gate_index + 1 (mod num_gates)
          4) Reward:
               - progress: decrease in distance-to-current-gate between steps
               - gate bonus when passed_gate
               - optional regularizers (control smoothness, altitude corridor, speed cap)
          5) Termination:
               - time limit: step_count >= max_episode_steps
               - safety: out of bounds in x/y/z and/or battery low
          6) Build next EnvState and next observation

        Hints:
          - Use jax.lax.select for branchless updates (JIT-friendly).
          - Keep all returned arrays as jnp.float32 where possible.
        """
        action = jnp.clip(action, -1.0, 1.0).astype(jnp.float32)

        x = state.x
        x_next = dynamics_step(x, action, SIM_DT, None)

        pos_prev = x[0:3]
        pos = x_next[0:3]
        vel = x_next[3:6]
        u_prev = x[16:20]

        gate_idx = state.gate_index.astype(jnp.int32)
        gate_pos, gate_quat = get_gate_pose(gate_idx)
        gate_forward = compute_gate_forward(gate_quat)

        pos_rel_prev = pos_prev - gate_pos
        pos_rel = pos - gate_pos

        forward_dist_prev = jnp.dot(pos_rel_prev, gate_forward)
        forward_dist = jnp.dot(pos_rel, gate_forward)
        lateral_dist = jnp.linalg.norm(pos_rel - forward_dist * gate_forward)

        crossed_plane = (forward_dist_prev <= 0.0) & (forward_dist > 0.0)
        within_radius = lateral_dist < params.gate_radius
        passed_gate = crossed_plane & within_radius
        missed_gate = crossed_plane & ~within_radius

        next_gate_idx = jax.lax.select(
            passed_gate,
            (gate_idx + 1) % NUM_GATES,
            gate_idx,
        )

        dist_to_gate = jnp.linalg.norm(pos_rel)
        next_gate_pos, _ = get_gate_pose(next_gate_idx)
        next_dist = jax.lax.select(
            passed_gate,
            jnp.linalg.norm(pos - next_gate_pos),
            dist_to_gate,
        )

        progress_reward = params.w_progress * (state.prev_dist_to_gate - next_dist)
        gate_reward = jax.lax.select(passed_gate, params.w_gate, 0.0)

        missed_gate_penalty = jax.lax.select(missed_gate, params.w_missed_gate, 0.0)
        control_penalty = params.w_control * jnp.sum((action - u_prev) ** 2)

        speed = jnp.linalg.norm(vel)
        excess_speed_penalty = params.w_speed * jnp.maximum(0.0, speed - MAX_SPEED) ** 2

        altitude = pos[2]
        altitude_low_penalty = params.w_altitude * jnp.maximum(0.0, 0.5 - altitude) ** 2
        altitude_high_penalty = params.w_altitude * jnp.maximum(0.0, altitude - 3.0) ** 2
        altitude_corridor_penalty = altitude_low_penalty + altitude_high_penalty

        out_of_bounds = (
            (pos[0] < BOUNDS_LOW[0])
            | (pos[0] > BOUNDS_HIGH[0])
            | (pos[1] < BOUNDS_LOW[1])
            | (pos[1] > BOUNDS_HIGH[1])
            | (pos[2] < MIN_ALTITUDE)
            | (pos[2] > BOUNDS_HIGH[2])
        )

        battery = x_next[20]
        battery_dead = battery < BATTERY_MIN

        time_limit = state.step_count >= params.max_episode_steps - 1

        done = out_of_bounds | battery_dead | time_limit

        survival_reward = jax.lax.select(done, 0.0, params.w_survival)
        crash_penalty = jax.lax.select(out_of_bounds & ~time_limit, params.w_crash, 0.0)
        timeout_penalty = jax.lax.select(time_limit, params.w_timeout, 0.0)

        reward = (
            progress_reward
            + gate_reward
            + survival_reward
            - missed_gate_penalty
            - control_penalty
            - excess_speed_penalty
            - altitude_corridor_penalty
            - crash_penalty
            - timeout_penalty
        )

        gates_passed = state.gates_passed + passed_gate.astype(jnp.float32)

        next_state = EnvState(
            x=x_next,
            gate_index=next_gate_idx.astype(jnp.float32),
            step_count=state.step_count + 1,
            prev_dist_to_gate=next_dist,
            gates_passed=gates_passed,
        )

        obs = self._obs_from_state(next_state)

        info = {
            "gates_passed": gates_passed,
            "out_of_bounds": out_of_bounds,
            "time_limit": time_limit.astype(jnp.float32),
            "dist_to_gate": dist_to_gate,
            "missed_gate": missed_gate.astype(jnp.float32),
        }

        return obs, next_state, reward, done, info


    def _obs_from_state(self, state: EnvState) -> jax.Array:
        """
        Construct observation from EnvState.

        TODO:
          - extract drone world state from x:
              pos_world = x[0:3]
              vel_world = x[3:6]
              q_world   = x[9:13]
              body_rates= x[13:16]
              u_prev    = x[16:20]
              battery   = x[-1]
          - get current gate pose (pos, quat) via get_gate_pose(gate_index)
          - compute:
              pos_gate = world_to_gate_position(pos_world, gate_pos, gate_quat)
              vel_gate = world_to_gate_vector(vel_world, gate_quat)
              q_rel    = quat_mul(quat_conjugate(gate_quat), q_world)
              euler_rel= quat_to_euler(q_rel)
          - next gate relative position (in current gate frame):
              next_gate_pos, _ = get_gate_pose((gate_index + 1) % num_gates)
              next_gate_rel = world_to_gate_vector(next_gate_pos - gate_pos, gate_quat)
          - concatenate into obs vector of shape (obs_size,)

        Optional (recommended):
          - include additional look-ahead gate(s)
          - add observation noise (Gaussian) in reset/step
        """
        x = state.x
        gate_idx = state.gate_index.astype(jnp.int32)

        pos_world = x[0:3]
        vel_world = x[3:6]
        q_world = x[9:13]
        body_rates = x[13:16]
        u_prev = x[16:20]
        battery = x[20:21]

        gate_pos, gate_quat = get_gate_pose(gate_idx)

        pos_gate = world_to_gate_position(pos_world, gate_pos, gate_quat)
        vel_gate = world_to_gate_vector(vel_world, gate_quat)

        q_gate_inv = quat_conjugate(gate_quat)
        q_rel = quat_mul(q_gate_inv, q_world)
        q_rel = quat_normalize(q_rel)
        euler_rel = quat_to_euler(q_rel)

        next_gate_idx = (gate_idx + 1) % NUM_GATES
        next_gate_pos, _ = get_gate_pose(next_gate_idx)
        next_gate_rel = world_to_gate_position(next_gate_pos, gate_pos, gate_quat)

        obs = jnp.concatenate(
            [
                pos_gate,
                vel_gate,
                euler_rel,
                body_rates,
                u_prev,
                battery,
                next_gate_rel,
            ]
        )

        return obs.astype(jnp.float32)


if __name__ == "__main__":
    env = DroneRaceEnv()
    params = DEFAULT_PARAMS
    key = jax.random.PRNGKey(42)

    obs, state = env.reset(key, params)
    print(f"Initial obs shape: {obs.shape}")
    print(f"Initial obs: {obs}")

    key, subkey = jax.random.split(key)
    action = jax.random.uniform(subkey, (4,), minval=-1.0, maxval=1.0)

    obs_next, state_next, reward, done, info = env.step(subkey, state, action, params)
    print(f"Next obs shape: {obs_next.shape}")
    print(f"Reward: {reward}, Done: {done}")
    print(f"Info: {info}")

    for i in range(100):
        key, subkey = jax.random.split(key)
        action = jax.random.uniform(subkey, (4,), minval=-0.5, maxval=0.5)
        action = action.at[3].set(0.0)
        obs, state, reward, done, info = env.step(subkey, state, action, params)

        if done:
            print(f"Episode ended at step {i}, gates passed: {info['gates_passed']}")
            break

    print("Environment test passed!")
