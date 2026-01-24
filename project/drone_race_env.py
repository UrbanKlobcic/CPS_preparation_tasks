"""
DroneRaceEnv

State/action conventions match the course handout.
Extensions:
- Action History Buffer (last 3 commands)
- Observation Noise
"""

from __future__ import annotations
from typing import NamedTuple
import jax
import jax.numpy as jnp
from utils import *
from gymnax.environments import spaces

# ---------------------------------------------------------------------
# Dynamics & constants
# ---------------------------------------------------------------------
def dynamics_step(x: jax.Array, u: jax.Array, dt: float, params) -> jax.Array:
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
    voltage = x[20] - 0.001 
    
    x_next = jnp.concatenate([
        pos_next, vel_next, acc, q_next, w_next, u_eff, jnp.array([voltage])
    ])
    return x_next

SIM_HZ = 100
SIM_DT = 1.0 / SIM_HZ
START_POS = jnp.array([18.6, 2.0, 1.5], dtype=jnp.float32)

ENVIRONMENT = jnp.array([
    [0, 12.5, 2.0, 1.35, -0.707107, 0.0, 0.0, 0.707107],
    [1, 6.5, 6.0, 1.35, -0.382684, 0.0, 0.0, 0.923879],
    [2, 5.5, 14.0, 1.35, -0.258819, 0.0, 0.0, 0.965926],
    [3, 2.5, 24.0, 1.35, 0.0, 0.0, 0.0, 1.0],
    [4, 7.5, 30.0, 1.35, -0.642788, 0.0, 0.0, 0.766044],
    [8, 18.5, 22.0, 1.35, -0.087155, 0.0, 0.0, 0.996195],
    [9, 20.5, 14.0, 1.35, 0.087155, 0.0, 0.0, 0.996195],
    [10, 18.5, 6.0, 1.35, 0.382684, 0.0, 0.0, 0.923879],
], dtype=jnp.float32)

MIN_POS = jnp.min(ENVIRONMENT[:, 1:4], axis=0)
MAX_POS = jnp.max(ENVIRONMENT[:, 1:4], axis=0)
BOUNDS_LOW = MIN_POS - jnp.array([5.0, 5.0, 2.0], dtype=jnp.float32)
BOUNDS_HIGH = MAX_POS + jnp.array([5.0, 5.0, 5.0], dtype=jnp.float32)
NUM_GATES = ENVIRONMENT.shape[0]
MIN_ALTITUDE = 0.0 
BATTERY_MIN = 22.0
BATTERY_MAX = 24.0

class EnvParams(NamedTuple):
    gate_radius: float = 1.0 
    max_episode_steps: int = SIM_HZ * 20
    # REWARD TUNING (Baseline)
    w_gate: float = 10.0
    w_progress: float = 1.0
    w_survival: float = 0.01
    w_control: float = 0.0
    w_altitude: float = 0.0
    w_speed: float = 0.001
    w_missed_gate: float = 1.0
    w_crash: float = 5.0
    w_timeout: float = 1.0
    
    # EXTENSION: NOISE PARAMETERS (Std Dev)
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
    # EXTENSION: Action History Buffer (3 steps x 4 dims)
    action_history: jax.Array 

class DroneRaceEnv:
    # 20 Base - 4 (u_prev) + 12 (history) = 28
    obs_size: int = 28
    action_size: int = 4

    @property
    def default_params(self) -> EnvParams:
        return DEFAULT_PARAMS

    def action_space(self, params=None):
        return spaces.Box(low=-1.0, high=1.0, shape=(4,))

    def observation_space(self, params=None):
        return spaces.Box(low=-1.0, high=1.0, shape=(self.obs_size,))

    def reset(self, rng: jax.Array, params) -> tuple[jax.Array, EnvState]:
        rng, rng_yaw, rng_obs = jax.random.split(rng, 3)
        gate_idx = jnp.array(0, dtype=jnp.int32)
        gate_pos, _ = get_gate_pose(gate_idx)
        
        init_pos = START_POS
        target_dir = gate_pos - init_pos
        target_yaw = jnp.arctan2(target_dir[1], target_dir[0])
        yaw_noise = jax.random.uniform(rng_yaw, (), minval=-0.2, maxval=0.2)
        init_quat = yaw_to_quat(target_yaw + yaw_noise)
        
        x0 = jnp.zeros(21, dtype=jnp.float32)
        x0 = x0.at[0:3].set(init_pos)
        x0 = x0.at[9:13].set(init_quat)
        x0 = x0.at[16:20].set(jnp.array([0.0, 0.0, 0.0, -0.5], dtype=jnp.float32))
        x0 = x0.at[20].set(BATTERY_MAX)

        init_dist = jnp.linalg.norm(init_pos - gate_pos)

        state = EnvState(
            x=x0,
            gate_index=gate_idx.astype(jnp.float32),
            step_count=jnp.array(0, dtype=jnp.float32),
            prev_dist_to_gate=init_dist,
            gates_passed=jnp.array(0, dtype=jnp.float32),
            action_history=jnp.zeros((3, 4), dtype=jnp.float32)
        )
        return self._obs_from_state(state, rng_obs, params), state

    def step(self, rng: jax.Array, state: EnvState, action: jax.Array, params: EnvParams):
        action = jnp.clip(action, -1.0, 1.0).astype(jnp.float32)
        x = state.x
        x_next = dynamics_step(x, action, SIM_DT, None)

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
        
        # Rewards
        progress_term = jax.lax.select(
            passed_gate, 
            state.prev_dist_to_gate,           
            state.prev_dist_to_gate - dist_to_current 
        )
        progress_reward = params.w_progress * progress_term
        
        out_of_bounds = (
            (pos[0] < BOUNDS_LOW[0]) | (pos[0] > BOUNDS_HIGH[0]) |
            (pos[1] < BOUNDS_LOW[1]) | (pos[1] > BOUNDS_HIGH[1]) |
            (pos[2] < MIN_ALTITUDE) | (pos[2] > BOUNDS_HIGH[2])
        )
        battery_dead = x_next[20] < BATTERY_MIN
        time_limit = state.step_count >= params.max_episode_steps - 1
        done = out_of_bounds | battery_dead | time_limit

        rew_components = (
            progress_reward +
            jax.lax.select(passed_gate, params.w_gate, 0.0) +
            jax.lax.select(done, 0.0, params.w_survival) -
            jax.lax.select(missed_gate, params.w_missed_gate, 0.0) -
            jax.lax.select(out_of_bounds & ~time_limit, params.w_crash, 0.0) -
            jax.lax.select(time_limit, params.w_timeout, 0.0)
        )
        
        dist_tracker = jax.lax.select(passed_gate, dist_to_next, dist_to_current)
        
        # EXTENSION: Update Action History
        # Shift: [t-1, t-2, t-3] -> [action, t-1, t-2]
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
        
        # Split RNG for Obs noise
        rng, rng_obs = jax.random.split(rng)
        
        return self._obs_from_state(next_state, rng_obs, params), next_state, rew_components, done, info

    def _obs_from_state(self, state: EnvState, key: jax.Array, params: EnvParams) -> jax.Array:
        x = state.x
        gate_idx = state.gate_index.astype(jnp.int32)
        gate_pos, gate_quat = get_gate_pose(gate_idx)

        # Baseline Frame Logic (Raw Gate Frame)
        pos_gate = world_to_gate_position(x[0:3], gate_pos, gate_quat)
        vel_gate = world_to_gate_vector(x[3:6], gate_quat)
        
        # EXTENSION: Add Noise before scaling
        # We need 3 keys for Pos, Vel, Rate
        k1, k2, k3, k4 = jax.random.split(key, 4)
        
        # Noise magnitude from params
        pos_gate = pos_gate + jax.random.normal(k1, shape=pos_gate.shape) * params.noise_pos
        vel_gate = vel_gate + jax.random.normal(k2, shape=vel_gate.shape) * params.noise_vel
        
        # Scaling
        pos_feat = pos_gate / 10.0
        vel_feat = vel_gate / 10.0
        
        q_rel = quat_mul(quat_conjugate(gate_quat), x[9:13])
        euler_rel = quat_to_euler(q_rel)
        # Add Noise to orientation
        euler_rel = euler_rel + jax.random.normal(k3, shape=euler_rel.shape) * params.noise_ori
        
        rates = x[13:16]
        # Add Noise to rates
        rates = rates + jax.random.normal(k4, shape=rates.shape) * params.noise_rate
        rates_feat = rates / 10.0 
        
        next_gate_idx = (gate_idx + 1) % NUM_GATES
        next_gate_pos, _ = get_gate_pose(next_gate_idx)
        # No noise on next gate position (known map)
        next_gate_rel = world_to_gate_position(next_gate_pos, gate_pos, gate_quat) / 10.0
        
        # Flatten History
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

if __name__ == "__main__":
    import numpy as np
    np.set_printoptions(precision=4, suppress=True)
    
    print("Running Tests for Extensions...")
    env = DroneRaceEnv()
    params = DEFAULT_PARAMS
    rng = jax.random.PRNGKey(42)
    
    obs, state = env.reset(rng, params)
    
    # Check Obs Size
    print(f"Observation Shape: {obs.shape}")
    assert obs.shape[0] == 28, f"Expected 28, got {obs.shape[0]}"
    
    # Check History Updating
    print("\nChecking Action History...")
    action1 = jnp.array([0.1, 0.2, 0.3, 0.4])
    obs1, state1, _, _, _ = env.step(rng, state, action1, params)
    
    print(f"History Step 1: {state1.action_history}")
    assert jnp.allclose(state1.action_history[0], action1)
    assert jnp.allclose(state1.action_history[1], jnp.zeros(4))
    
    action2 = jnp.array([-0.5, -0.5, -0.5, -0.5])
    obs2, state2, _, _, _ = env.step(rng, state1, action2, params)
    
    print(f"History Step 2: {state2.action_history}")
    assert jnp.allclose(state2.action_history[0], action2)
    assert jnp.allclose(state2.action_history[1], action1)
    
    # Check Noise
    print("\nChecking Noise (Deterministic State)...")
    # Reset to same state
    state_fixed = state._replace()
    rng_noise = jax.random.PRNGKey(1)
    
    # Call obs twice with different RNG
    rng_a, rng_b = jax.random.split(rng_noise)
    obs_a = env._obs_from_state(state_fixed, rng_a, params)
    obs_b = env._obs_from_state(state_fixed, rng_b, params)
    
    diff = jnp.linalg.norm(obs_a - obs_b)
    print(f"Obs Diff (should be > 0): {diff}")
    assert diff > 1e-4
    
    print("\n\033[92mALL EXTENSION TESTS PASSED\033[0m")
