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

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
from utils import *  # import quaternion helpers from utils.py
from gymnax.environments import spaces


# ---------------------------------------------------------------------
# TODO: import your learned dynamics model from Task 1
# ---------------------------------------------------------------------
# Your Task-1 model should have signature:
#   x_next = step(x, u, dt, params)
#
# Example (adjust to your project structure):
# from envs.models.dynamics import step as dynamics_step, DEFAULT_PARAMS, ModelParameters
#
def dynamics_step(x: jax.Array, u: jax.Array, dt: float, params) -> jax.Array:
    raise NotImplementedError(
        "TODO: import and use your Task-1 learned dynamics step(x,u,dt,params)"
    )


# Constants: simulation + track
SIM_HZ = 100
SIM_DT = 1.0 / SIM_HZ

START_POS = jnp.array([18.6, 2.0, 0.1], dtype=jnp.float32)

# Racing gates: [id, x, y, z, qw, qx, qy, qz]
# Note: z is 1.35 for all gates as this is the middle of the gate (e.g. 2.7 / 2)
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


# Gate-frame utilities
def get_gate_pose(gate_index: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Return (gate_pos, gate_quat) for the given index."""
    idx = gate_index.astype(jnp.int32)
    gate_row = ENVIRONMENT[idx]
    pos = gate_row[1:4]
    quat = gate_row[4:8]
    quat = quat / (jnp.linalg.norm(quat) + 1e-8)
    return pos, quat


def world_to_gate_position(
        pos_world: jax.Array, gate_pos: jax.Array, gate_quat: jax.Array
) -> jax.Array:
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


@dataclass(frozen=True)
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

    gate_radius: float = 0.4
    gate_bonus: float = 50.0
    max_episode_steps: int = SIM_HZ * 12

    # Reward weights (examples for reward shaping)
    w_progress: float = 1.0
    w_control: float = 0.01
    w_altitude: float = 0.02
    w_speed: float = 0.01

    # Observation layout target for the baseline implementation:
    #   pos_gate(3) + vel_gate(3) + euler_rel(3) + body_rates(3) + u_prev(4) + battery(1) + next_gate_rel(3) = 20
    obs_size: int = 20

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
        raise NotImplementedError("TODO: implement reset()")

    def step(self, rng: jax.Array, state: EnvState, action: jax.Array, params):
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
        raise NotImplementedError("TODO: implement step()")

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
        raise NotImplementedError("TODO: implement _obs_from_state()")


if __name__ == "__main__":
    env = DroneRaceEnv()
    key = jax.random.PRNGKey(0)
    # This will raise until you implement reset().
    obs, s = env.reset(key, params=None)
    print("obs shape:", obs.shape)
