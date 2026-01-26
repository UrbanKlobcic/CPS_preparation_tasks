import jax
import jax.numpy as jnp
import chex
import numpy as np
from flax import struct
from functools import partial
from typing import Optional, Tuple, Union, Any
from gymnax.environments import environment, spaces

class GymnaxWrapper(object):
    def __init__(self, env):
        self._env = env
    def __getattr__(self, name):
        return getattr(self._env, name)

# --- 1. Vectorization Wrapper ---
class VecEnv(GymnaxWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.reset = jax.vmap(self._env.reset, in_axes=(0, None))
        self.step = jax.vmap(self._env.step, in_axes=(0, 0, 0, None))

    def observation_space(self, params):
        return self._env.observation_space(params)

    def action_space(self, params):
        return self._env.action_space(params)

# --- 2. Auto Reset Wrapper ---
class AutoResetWrapper(GymnaxWrapper):
    """Automatically resets the environment when done=True."""
    def __init__(self, env):
        super().__init__(env)

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state, action, params=None):
        obs, next_state, reward, done, info = self._env.step(key, state, action, params)
        
        # Split RNG for potential reset
        key, reset_key = jax.random.split(key)
        
        # Calculate reset state/obs
        reset_obs, reset_state = self._env.reset(reset_key, params)
        
        # Select between next_state and reset_state based on done
        new_state = jax.tree.map(
            lambda x, y: jnp.where(done, x, y), 
            reset_state, 
            next_state
        )
        
        done_expanded = done
        if obs.ndim > 0 and done.ndim > 0 and done.shape != obs.shape:
             done_expanded = done[..., None] # Broadcast if needed
             
        new_obs = jnp.where(done_expanded, reset_obs, obs)
        
        return new_obs, new_state, reward, done, info

# --- 3. Existing Wrappers ---
class FlattenObservationWrapper(GymnaxWrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def observation_space(self, params) -> spaces.Box:
        shape = (np.prod(self._env.observation_space(params).shape),)
        return spaces.Box(low=-1.0, high=1.0, shape=shape, dtype=jnp.float32)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key, params=None):
        obs, state = self._env.reset(key, params)
        obs = jnp.reshape(obs, (-1,))
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state, action, params=None):
        obs, state, reward, done, info = self._env.step(key, state, action, params)
        obs = jnp.reshape(obs, (-1,))
        return obs, state, reward, done, info

@struct.dataclass
class LogEnvState:
    env_state: Any
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_lengths: int
    timestep: int

class LogWrapper(GymnaxWrapper):
    def __init__(self, env):
        super().__init__(env)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key, params=None):
        obs, env_state = self._env.reset(key, params)
        state = LogEnvState(env_state, 0.0, 0, 0.0, 0, 0)
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state, action, params=None):
        obs, env_state, reward, done, info = self._env.step(key, state.env_state, action, params)
        new_episode_return = state.episode_returns + reward
        new_episode_length = state.episode_lengths + 1
        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - done),
            episode_lengths=new_episode_length * (1 - done),
            returned_episode_returns=state.returned_episode_returns * (1 - done) + new_episode_return * done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - done) + new_episode_length * done,
            timestep=state.timestep + 1,
        )
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["timestep"] = state.timestep
        info["returned_episode"] = done
        return obs, state, reward, done, info
