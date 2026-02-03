import io
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from drone_race_env import DroneRaceEnv, DEFAULT_PARAMS, ENVIRONMENT, GATE_WIDTH_OUTER, SIM_HZ
from wrappers import FlattenObservationWrapper, VecEnv, AutoResetWrapper
from network import ActorCritic
from utils import quat_rotate

NUM_ENVS = 256
STEPS = 500

def plot_density(network_params):
    env = DroneRaceEnv()
    
    # zero noise for evaluation + start at random gate
    env_params = DEFAULT_PARAMS._replace(
        max_episode_steps=SIM_HZ * 60,
        noise_pos=0.0,
        noise_vel=0.0,
        noise_ori=0.0,
        noise_rate=0.0
    )
    
    env = FlattenObservationWrapper(env)
    env = AutoResetWrapper(env)
    env = VecEnv(env)
    
    network = ActorCritic(env.action_space(env_params).shape[0], activation="tanh")
    
    rng = jax.random.PRNGKey(0)
    rng, _rng = jax.random.split(rng)
    init_x = jnp.zeros(env.observation_space(env_params).shape)
    network.init(_rng, init_x)
    
    print(f"Generating density plot from {NUM_ENVS} envs...")
    
    rng, reset_key = jax.random.split(rng)
    reset_keys = jax.random.split(reset_key, NUM_ENVS)
    obs, state = env.reset(reset_keys, env_params)

    all_x = []
    all_y = []

    @jax.jit
    def step_fn(params, obs, state, rng):
        rng, action_key, step_key = jax.random.split(rng, 3)
        
        pi, _ = network.apply(params, obs)
        
        action = pi.sample(seed=action_key)
        
        step_keys = jax.random.split(step_key, NUM_ENVS)
        next_obs, next_state, _, _, _ = env.step(step_keys, state, action, env_params)
        
        raw_x = next_state.x[..., 0]
        raw_y = next_state.x[..., 1]
        
        return next_obs, next_state, raw_x, raw_y, rng

    for _ in range(STEPS):
        obs, state, batch_x, batch_y, rng = step_fn(network_params, obs, state, rng)
        all_x.append(np.array(batch_x))
        all_y.append(np.array(batch_y))

    flat_x = np.concatenate(all_x).flatten()
    flat_y = np.concatenate(all_y).flatten()

    print("Plotting...")
    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    ax.set_facecolor('#200030')
    ax.hist2d(flat_x, flat_y, bins=150, cmap='viridis', range=[[0, 25], [0, 35]])
    
    for gate in ENVIRONMENT:
        pos = gate[1:4]
        quat = gate[4:8]
        
        local_dir = jnp.array([1.0, 0.0, 0.0])
        world_dir = quat_rotate(quat, local_dir)
        
        cx, cy = float(pos[0]), float(pos[1])
        dx, dy = float(world_dir[0]), float(world_dir[1])
        
        x1 = cx - dx * (GATE_WIDTH_OUTER / 2)
        y1 = cy - dy * (GATE_WIDTH_OUTER / 2)
        x2 = cx + dx * (GATE_WIDTH_OUTER / 2)
        y2 = cy + dy * (GATE_WIDTH_OUTER / 2)
        
        ax.plot([x1, x2], [y1, y2], color='white', linewidth=2, solid_capstyle='round')
    
    ax.set_aspect('equal')
    ax.set_title(f"Position Density")
    ax.set_xlabel("x (world)")
    ax.set_ylabel("y (world)")
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    plt.close(fig)
    return buf

