"""
DroneRaceEnv

This environment has following functions:
- reset() and step() are pure functions (no side effects)
- suitable for vectorization (vmap) and JIT compilation
- designed to be used with a pure-JAX PPO implementation

Implemented:
- reward shaping
- termination conditions
- optional observation noise and action-history features

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
from typing import NamedTuple, Optional, Callable
import jax
import jax.numpy as jnp
from utils import *
from gymnax.environments import spaces

# ---------------------------------------------------------------------
# TODO: Use Task 1 model here
# Dynamics & constants
# ---------------------------------------------------------------------
def dynamics_step(x: jax.Array, u: jax.Array, dt: float) -> jax.Array:
    m = 1.0
    g = 9.80665
    max_rates = jnp.array([10.786, 10.786, 2.094]) 
    
    alpha = 0.6 
    u_prev = x[16:20]
    u_eff = u_prev * (1 - alpha) + u * alpha
    
    w_target = u_eff[0:3] * max_rates
    w_next = w_target 

    q = x[9:13]
    w_quat = jnp.array([0.0, w_next[0], w_next[1], w_next[2]])
    dq = 0.5 * quat_mul(q, w_quat)
    q_next = quat_normalize(q + dq * dt)
    
    throttle = (u_eff[3] + 1.0) / 2.0
    thrust_mag = throttle * 40.0
    
    thrust_local = jnp.array([0.0, 0.0, thrust_mag])
    thrust_world = quat_rotate(q_next, thrust_local)
    
    acc = thrust_world / m - jnp.array([0.0, 0.0, g])
    vel = x[3:6]
    drag = -0.1 * vel
    acc = acc + drag
    
    vel_next = vel + acc * dt
    pos_next = x[0:3] + vel * dt + 0.5 * acc * dt**2
    
    discharge_rate_per_sec = 1.0 / 60.0 
    voltage = x[20] - (discharge_rate_per_sec * dt)
    
    x_next = jnp.concatenate([
        pos_next, vel_next, acc, q_next, w_next, u_eff, jnp.array([voltage])
    ])
    return x_next


SIM_HZ = 100
SIM_DT = 1.0 / SIM_HZ
START_POS = jnp.array([18.6, 2.0, 0.1], dtype=jnp.float32)

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
BATTERY_MIN = 22.0
BATTERY_MAX = 24.0

class EnvParams(NamedTuple):
    gate_radius: float = 0.75
    max_episode_steps: int = SIM_HZ * 40
    
    # Evaluation config: -1 for random, >=0 for specific gate
    initial_gate_id: int = -1
    
    # rewards
    w_gate: float = 100.0
    w_progress: float = 0.5
    w_speed: float = 0.01
    w_survival: float = -0.01
    
    # penalties
    w_control: float = 0.001
    w_altitude: float = 0.001
    w_missed_gate: float = 0.5
    w_crash: float = 100.0
    w_timeout: float = 0.0
    
    # EXTENSION: noise parameters (std dev)
    noise_pos: float = 0.05
    noise_vel: float = 0.1
    noise_ori: float = 0.02
    noise_rate: float = 0.05

DEFAULT_PARAMS = EnvParams()

def compute_gate_forward(gate_quat: jax.Array) -> jax.Array:
    return quat_rotate(gate_quat, jnp.array([0.0, -1.0, 0.0], dtype=jnp.float32))

def get_gate_pose(gate_index: jax.Array) -> tuple[jax.Array, jax.Array]:
    idx = gate_index.astype(jnp.int32)
    gate_row = ENVIRONMENT[idx]
    pos = gate_row[1:4]
    quat = gate_row[4:8]
    quat = quat / (jnp.linalg.norm(quat) + 1e-8)
    return pos, quat

def world_to_gate_position(pos_world, gate_pos, gate_quat):
    q_inv = quat_conjugate(gate_quat)
    return quat_rotate(q_inv, pos_world - gate_pos)

def world_to_gate_vector(vec_world, gate_quat):
    q_inv = quat_conjugate(gate_quat)
    return quat_rotate(q_inv, vec_world)

class EnvState(NamedTuple):
    x: jax.Array
    gate_index: jax.Array
    step_count: jax.Array
    prev_dist_to_gate: jax.Array
    gates_passed: jax.Array
    # EXTENSION: action history buffer (last 3 steps)
    action_history: jax.Array 

class DroneRaceEnv:
    """
    Drone racing environment.

    Implemented:
      - observation in *current gate frame*
      - observation includes *relative position of the next gate*
      - reward encourages progressing through gates
      - termination on time limit and safety bounds
      - action history buffer
      - observation noise
    """
    obs_size: int = 28
    action_size: int = 4

    def __init__(self, dynamics_fn: Optional[Callable] = None):
        """
        Args:
            dynamics_fn: Function (x, u) -> x_next. 
                         Defaults to internal `dynamics_step`.
        """
        self.dynamics_fn = dynamics_fn if dynamics_fn is not None else dynamics_step

    @property
    def default_params(self) -> EnvParams:
        return DEFAULT_PARAMS

    def action_space(self, params=None):
        return spaces.Box(low=-1.0, high=1.0, shape=(4,))

    def observation_space(self, params=None):
        return spaces.Box(low=-1.0, high=1.0, shape=(self.obs_size,))

    def reset(self, rng: jax.Array, params) -> tuple[jax.Array, EnvState]:
        """
        Reset environment and return (obs, state).
        
        Implemented:
          1) Create an initial drone state x0, shape (21,):
             - position: START_POS
             - orientation: face gate 0 (use yaw_to_quat and a look-at direction)
             - battery voltage: e.g., 24.0
             - optionally set velocity to 0 and previous action to 0
          2) Set gate_index = 0, step_count = 0
          3) Compute observation in current gate frame via _obs_from_state()
        
        Implemented (Optional):
          - randomize initial positions/orientation slightly (robustness)
          - do not always start at gate 0 but slightly after a random gate (strongly recommended)
        """
        rng, key_gate, key_pos, key_yaw, key_obs = jax.random.split(rng, 5)
        
        # random initial gate id unless specified
        rand_gate = jax.random.randint(key_gate, (), 0, NUM_GATES).astype(jnp.int32)
        target_gate_idx = jax.lax.select(
            params.initial_gate_id == -1, 
            rand_gate, 
            jnp.array(params.initial_gate_id, dtype=jnp.int32)
        ) 
        
        prev_gate_idx = (target_gate_idx - 1) % NUM_GATES
        prev_gate_pos, _ = get_gate_pose(prev_gate_idx)
        
        source_pos = jax.lax.select(
            target_gate_idx == 0,
            START_POS,
            prev_gate_pos
        )
        
        target_pos, _ = get_gate_pose(target_gate_idx)
        
        vec = target_pos - source_pos
        dist = jnp.linalg.norm(vec)
        direction = vec / (dist + 1e-6)
        
        base_pos = source_pos + direction * 2.0
        
        pos_noise = jax.random.uniform(key_pos, (3,), minval=-1.0, maxval=1.0)
        pos_noise = pos_noise * jnp.array([1.0, 1.0, 0.5])
        
        init_pos = base_pos + pos_noise
        init_pos = init_pos.at[2].set(jnp.maximum(init_pos[2], 0.5))
        
        ideal_yaw = jnp.arctan2(direction[1], direction[0])
        yaw_noise = jax.random.uniform(key_yaw, (), minval=-0.5, maxval=0.5)
        init_quat = yaw_to_quat(ideal_yaw + yaw_noise)
        
        x0 = jnp.zeros(21, dtype=jnp.float32)
        x0 = x0.at[0:3].set(init_pos)
        x0 = x0.at[9:13].set(init_quat)
        x0 = x0.at[16:20].set(jnp.array([0.0, 0.0, 0.0, -0.6], dtype=jnp.float32))
        x0 = x0.at[20].set(BATTERY_MAX)
        
        init_dist_to_target = jnp.linalg.norm(init_pos - target_pos)
        
        state = EnvState(
            x=x0,
            gate_index=target_gate_idx.astype(jnp.float32),
            step_count=jnp.array(0, dtype=jnp.float32),
            prev_dist_to_gate=init_dist_to_target,
            gates_passed=jnp.array(0, dtype=jnp.float32),
            action_history=jnp.zeros((3, 4), dtype=jnp.float32)
        )
        return self._obs_from_state(state, key_obs, params), state

    def step(self, rng: jax.Array, state: EnvState, action: jax.Array, params: EnvParams):
        """
        Step the environment and return (obs, next_state, reward, done, info).
        
        Implemented:
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
               - regularizers (survival, speed, control smoothness, altitude corridor)
          5) Termination:
               - time limit: step_count >= max_episode_steps
               - safety: out of bounds in x/y/z and/or battery low
          6) Action history extension:
               - action history buffer (last 3 actions)
          7) Build next EnvState and next observation
        """
        action = jnp.clip(action, -1.0, 1.0).astype(jnp.float32)
        x = state.x
        x_next = self.dynamics_fn(x, action, SIM_DT)
        
        pos = x_next[0:3]
        vel = x_next[3:6]
        
        gate_idx = state.gate_index.astype(jnp.int32)
        gate_pos, gate_quat = get_gate_pose(gate_idx)
        gate_forward = compute_gate_forward(gate_quat)
        
        pos_rel_prev = state.x[0:3] - gate_pos
        pos_rel = pos - gate_pos
        
        forward_dist_prev = jnp.dot(pos_rel_prev, gate_forward)
        forward_dist = jnp.dot(pos_rel, gate_forward)
        lateral_dist = jnp.linalg.norm(pos_rel - forward_dist * gate_forward)
        
        # Gate Logic
        crossed_plane = (forward_dist_prev <= 0.0) & (forward_dist > 0.0)
        within_radius = lateral_dist < params.gate_radius
        passed_gate = crossed_plane & within_radius
        missed_gate = crossed_plane & ~within_radius
        
        next_gate_idx = jax.lax.select(passed_gate, (gate_idx + 1) % NUM_GATES, gate_idx)
        dist_to_current = jnp.linalg.norm(pos_rel)
        next_gate_pos, _ = get_gate_pose(next_gate_idx)
        dist_to_next = jnp.linalg.norm(pos - next_gate_pos)
        
        # Out of Bounds / Timeout
        out_of_bounds = (
            (pos[0] < BOUNDS[0, 0]) | (pos[0] > BOUNDS[1, 0]) |
            (pos[1] < BOUNDS[0, 1]) | (pos[1] > BOUNDS[1, 1]) |
            (pos[2] < BOUNDS[0, 2]) | (pos[2] > BOUNDS[1, 2])
        )
        battery_dead = x_next[20] < BATTERY_MIN
        time_limit = state.step_count >= params.max_episode_steps - 1
        done = out_of_bounds | battery_dead | time_limit
        
        # Rewards / Penalties
        progress_term = jax.lax.select(
            passed_gate,
            state.prev_dist_to_gate,
            state.prev_dist_to_gate - dist_to_current 
        )
        r_progress = params.w_progress * progress_term
        r_gate = jax.lax.select(passed_gate, params.w_gate, 0.0)
        r_survival = jax.lax.select(done, 0.0, params.w_survival)
        r_speed = params.w_speed * jnp.linalg.norm(vel)
        
        p_missed = jax.lax.select(missed_gate, params.w_missed_gate, 0.0)
        p_crash = jax.lax.select(out_of_bounds & ~time_limit, params.w_crash, 0.0)
        p_timeout = jax.lax.select(time_limit, params.w_timeout, 0.0)
        p_control = params.w_control * jnp.linalg.norm(action - state.action_history[0])
        p_altitude = params.w_altitude * jnp.abs(pos[2] - gate_pos[2])
        
        reward = (
            r_progress +
            r_gate +
            r_survival +
            r_speed -
            p_missed -
            p_crash -
            p_timeout -
            p_control -
            p_altitude
        )
        
        dist_tracker = jax.lax.select(passed_gate, dist_to_next, dist_to_current)
        
        # EXTENSION: update action history
        new_history = jnp.concatenate([action[None, :], state.action_history[:-1]], axis=0)
        
        next_state = EnvState(
            x=x_next,
            gate_index=next_gate_idx.astype(jnp.float32),
            step_count=state.step_count + 1,
            prev_dist_to_gate=dist_tracker,
            gates_passed=state.gates_passed + passed_gate.astype(jnp.float32),
            action_history=new_history
        )
        
        info = {
            "gates_passed": next_state.gates_passed,
            "out_of_bounds": out_of_bounds,
            "returned_episode": done, 
            "timestep": state.step_count,
            "pos_z": pos[2] 
        }
        
        rng, rng_obs = jax.random.split(rng)
        return self._obs_from_state(next_state, rng_obs, params), next_state, reward, done, info

    def _obs_from_state(self, state: EnvState, key: jax.Array, params: EnvParams) -> jax.Array:
        """
        Construct observation from EnvState.
        
        Implemented:
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
          - EXTENSION: normalize larger values
          - EXTENSION: add gaussian noise to pos/vel/euler_rel/rates
        """
        x = state.x
        gate_idx = state.gate_index.astype(jnp.int32)
        gate_pos, gate_quat = get_gate_pose(gate_idx)
        
        pos_gate = world_to_gate_position(x[0:3], gate_pos, gate_quat)
        vel_gate = world_to_gate_vector(x[3:6], gate_quat)
        
        k1, k2, k3, k4 = jax.random.split(key, 4)
        
        # Noise magnitude from params
        pos_gate = pos_gate + jax.random.normal(k1, shape=pos_gate.shape) * params.noise_pos
        vel_gate = vel_gate + jax.random.normal(k2, shape=vel_gate.shape) * params.noise_vel
        
        # normalize large values
        pos_feat = pos_gate / 10.0
        vel_feat = vel_gate / 10.0
        
        q_rel = quat_mul(quat_conjugate(gate_quat), x[9:13])
        euler_rel = quat_to_euler(q_rel)
        # orientation noise
        euler_rel = euler_rel + jax.random.normal(k3, shape=euler_rel.shape) * params.noise_ori
        
        rates = x[13:16]
        # rates noise
        rates = rates + jax.random.normal(k4, shape=rates.shape) * params.noise_rate
        rates_feat = rates / 10.0 
        
        next_gate_idx = (gate_idx + 1) % NUM_GATES
        next_gate_pos, _ = get_gate_pose(next_gate_idx)
        next_gate_rel = world_to_gate_position(next_gate_pos, gate_pos, gate_quat) / 10.0
        
        hist_feat = state.action_history.reshape(-1)
        
        # 3+3+3+3+12+1+3 = 28
        obs = jnp.concatenate([
            pos_feat, 
            vel_feat, 
            euler_rel, 
            rates_feat, 
            hist_feat, 
            x[20:21], 
            next_gate_rel
        ])
        return obs

