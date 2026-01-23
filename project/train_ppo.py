"""
Trains a PPO agent on the drone env.

Based on purejaxrl reference implementation:
https://github.com/luchris429/purejaxrl/blob/main/purejaxrl/ppo.py
"""
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
import distrax
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from typing import Any, NamedTuple, Tuple

from drone_race_env import DEFAULT_PARAMS, DroneRaceEnv, EnvParams, EnvState

# -----------------------------------------------------------------------------
# Wrappers
# -----------------------------------------------------------------------------

class AutoResetWrapper:
    """Resets the environment automatically when done is True."""
    def __init__(self, env: DroneRaceEnv):
        self.env = env
        self.obs_size = env.obs_size
        self.action_size = env.action_size

    def reset(self, rng, params: EnvParams) -> Tuple[jnp.ndarray, EnvState]:
        obs, state = self.env.reset(rng, params)
        assert obs.shape == (self.obs_size,), f"Reset obs shape mismatch: {obs.shape}"
        return obs, state

    def step(
        self, rng, state: EnvState, action: jnp.ndarray, params: EnvParams
    ) -> Tuple[jnp.ndarray, EnvState, jnp.ndarray, jnp.ndarray, dict]:

        obs, env_state, reward, done, info = self.env.step(rng, state, action, params)
        
        rng_reset = jax.random.split(rng)[0]
        obs_reset, env_state_reset = self.env.reset(rng_reset, params)
        
        assert obs.shape == obs_reset.shape, "Observation shapes must match"
        assert obs.dtype == obs_reset.dtype, "Observation dtypes must match"

        new_obs = jax.lax.select(done, obs_reset, obs)
        new_state = jax.tree.map(
            lambda x, y: jax.lax.select(done, x, y), 
            env_state_reset, 
            env_state
        )
        
        return new_obs, new_state, reward, done, info


class LogEnvState(NamedTuple):
    env_state: EnvState
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_lengths: int
    returned_oob: float


class LogWrapper:
    def __init__(self, env):
        self.env = env
        self.obs_size = env.obs_size
        self.action_size = env.action_size

    def reset(self, rng, params):
        obs, env_state = self.env.reset(rng, params)
        state = LogEnvState(
            env_state, 0., 0, 0., 0, 0.
        )
        return obs, state

    def step(self, rng, state, action, params):
        obs, env_state, reward, done, info = self.env.step(rng, state.env_state, action, params)
        
        new_return = state.episode_returns + reward
        new_length = state.episode_lengths + 1
        
        returned_returns = jax.lax.select(done, new_return, state.returned_episode_returns)
        returned_lengths = jax.lax.select(done, new_length, state.returned_episode_lengths)

        is_oob = info["out_of_bounds"]
        returned_oob = jax.lax.select(is_oob, 1., state.returned_oob)
        
        new_return = jax.lax.select(done, 0.0, new_return)
        new_length = jax.lax.select(done, 0, new_length)
        
        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_return,
            episode_lengths=new_length,
            returned_episode_returns=returned_returns,
            returned_episode_lengths=returned_lengths,
            returned_oob=returned_oob,
        )
        info["returned_episode_returns"] = returned_returns
        info["returned_episode_lengths"] = returned_lengths
        info["returned_episode"] = done
        info["returned_oob"] = returned_oob

        assert isinstance(state, LogEnvState), "State type mismatch"
        return obs, state, reward, done, info


class ObsNormState(NamedTuple):
    mean: jax.Array
    var: jax.Array
    env_state: Any

class ObsNormWrapper:
    """
    Normalize observations using exponential moving average.
    """

    def __init__(self, env):
        self.env = env
        self.obs_size = env.obs_size
        self.action_size = env.action_size
        self.alpha = 1e-3
        self.epsilon = 1e-8

    def reset(self, rng, params: EnvParams) -> Tuple[jnp.ndarray, ObsNormState]:
        obs, env_state = self.env.reset(rng, params)
        state = ObsNormState(
            mean=jnp.zeros(self.obs_size),
            var=jnp.ones(self.obs_size),
            env_state=env_state
        )
        assert state.mean.shape == (self.obs_size,), "Mean shape mismatch"
        assert state.var.shape == (self.obs_size,), "Var shape mismatch"
        
        return self._normalize(obs, state), state

    def step(self, rng, state, action, params):
        obs, env_state, reward, done, info = self.env.step(rng, state.env_state, action, params)
        
        new_mean = (1.0 - self.alpha) * state.mean + self.alpha * obs
        new_var = (1.0 - self.alpha) * state.var + self.alpha * jnp.square(obs - state.mean)
        
        state = ObsNormState(
            mean=new_mean,
            var=new_var,
            env_state=env_state
        )
        
        normalized_obs = self._normalize(obs, state)
        assert normalized_obs.shape == obs.shape, "Normalized obs shape mismatch"
        
        return normalized_obs, state, reward, done, info

    def _normalize(self, obs, state):
        return (obs - state.mean) / jnp.sqrt(state.var + self.epsilon)

# -----------------------------------------------------------------------------
# Network
# -----------------------------------------------------------------------------

class ActorCritic(nn.Module):
    action_dim: int
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        assert x.shape[-1] == 20, f"Input dim expected 20, got {x.shape[-1]}"
        
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        actor_mean = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(actor_mean)
        actor_mean = activation(actor_mean)
        
        actor_bias_init = jnp.zeros(self.action_dim).at[3].set(-0.5)
        actor_mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), 
                              bias_init=constant(actor_bias_init))(actor_mean)
        
        log_std = self.param("log_std", constant(-0.5), (self.action_dim,))
        
        pi = distrax.MultivariateNormalDiag(loc=actor_mean, scale_diag=jnp.exp(log_std))
        assert pi.event_shape == (self.action_dim,), f"Actor output shape mismatch: {pi.event_shape}"

        critic = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        critic = activation(critic)
        critic = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)

        return pi, jnp.squeeze(critic, axis=-1)

# -----------------------------------------------------------------------------
# Training Logic
# -----------------------------------------------------------------------------

def make_train(config):
    assert config["NUM_ENVS"] > 0, "NUM_ENVS must be positive"
    assert config["NUM_STEPS"] > 0, "NUM_STEPS must be positive"
    assert config["TOTAL_TIMESTEPS"] > 0
    assert config["LR"] > 0
    
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    config["MINIBATCH_SIZE"] = config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    
    assert config["NUM_UPDATES"] > 0, "Not enough timesteps for a single update"
    
    env_params = DEFAULT_PARAMS
    
    env = DroneRaceEnv()
    env = AutoResetWrapper(env)
    env = ObsNormWrapper(env)
    env = LogWrapper(env)

    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac

    def train(rng):
        network = ActorCritic(env.action_size, activation=config["ACTIVATION"])
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros((1, env.obs_size))
        network_params = network.init(_rng, init_x)
        
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
            
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)
        
        assert obsv.shape == (config["NUM_ENVS"], env.obs_size), "Init obs shape mismatch"

        def _update_step(runner_state, update_i):
            
            def _env_step(runner_state, _):
                train_state, env_state, last_obs, rng = runner_state
                
                rng, _rng = jax.random.split(rng)
                pi, value = network.apply(train_state.params, last_obs)
                
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                
                assert action.shape == (config["NUM_ENVS"], env.action_size)
                
                action_clipped = jnp.clip(action, -1.0, 1.0)

                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0, None)
                )(rng_step, env_state, action_clipped, env_params)
                
                assert obsv.shape == last_obs.shape
                assert reward.shape == (config["NUM_ENVS"],)
                
                transition = Transition(
                    done, action, value, reward, log_prob, last_obs, info
                )
                runner_state = (train_state, env_state, obsv, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )
            
            assert traj_batch.obs.shape[0] == config["NUM_STEPS"]

            train_state, env_state, last_obs, rng = runner_state
            _, last_val = network.apply(train_state.params, last_obs)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = transition.done, transition.value, transition.reward
                    
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)
            assert advantages.shape == (config["NUM_STEPS"], config["NUM_ENVS"])

            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        pi, value = network.apply(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)
                        entropy = pi.entropy().mean()

                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = jnp.clip(
                            ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]
                        ) * gae
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        assert total_loss.ndim == 0, "Total loss must be scalar"
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    (total_loss, (v_loss, p_loss, ent)), grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, (total_loss, v_loss, p_loss, ent)

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                
                batch = jax.tree.map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree.map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                
                minibatches = jax.tree.map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                
                assert minibatches[0].obs.shape[0] == config["NUM_MINIBATCHES"]

                train_state, losses = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, losses

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            
            train_state = update_state[0]
            rng = update_state[-1]
            loss_total, loss_v, loss_p, loss_ent = loss_info
            
            metric = {
                "total_loss": loss_total.mean(),
                "value_loss": loss_v.mean(),
                "policy_loss": loss_p.mean(),
                "entropy": loss_ent.mean(),
                "returned_episode_returns": traj_batch.info["returned_episode_returns"],
                "returned_episode_lengths": traj_batch.info["returned_episode_lengths"],
                "returned_oob": traj_batch.info["returned_oob"],
                "returned_episode": traj_batch.info["returned_episode"],
                "gates_passed": traj_batch.info["gates_passed"].mean(),
            }

            if config.get("DEBUG"):
                def callback(m, step):
                    step_val = int(step)
                    
                    mask = np.array(m["returned_episode"])
                    
                    if np.any(mask):
                        ret_arr = np.array(m["returned_episode_returns"])
                        len_arr = np.array(m["returned_episode_lengths"])
                        oob_arr = np.array(m["returned_oob"])
                        
                        avg_ret = np.sum(ret_arr * mask) / np.sum(mask)
                        avg_len = np.sum(len_arr * mask) / np.sum(mask)
                        crash_rate = np.sum(oob_arr * mask) / np.sum(mask)
                        gates = float(m['gates_passed'])
                        
                        print(
                            f"Step {step_val:<3} | "
                            f"Reward={avg_ret:<6.1f} | "
                            f"Len={avg_len:<6.1f} | "
                            f"Crash={crash_rate*100:.0f}% | "
                            f"Gates={gates:.2f}"
                        )
                
                jax.debug.callback(callback, metric, update_i)

            runner_state = (train_state, env_state, last_obs, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, _rng)
        
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, jnp.arange(config["NUM_UPDATES"]), config["NUM_UPDATES"]
        )
        return runner_state[0], metric

    return train


class Transition(NamedTuple):
    done: jax.Array
    action: jax.Array
    value: jax.Array
    reward: jax.Array
    log_prob: jax.Array
    obs: jax.Array
    info: dict


if __name__ == "__main__":
    config = {
        "LR": 3e-4,
        "NUM_ENVS": 64,
        "NUM_STEPS": 512,
        "TOTAL_TIMESTEPS": 20_000_000,
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 4,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ACTIVATION": "tanh",
        "ANNEAL_LR": True,
        "DEBUG": True,
        "CKPT_DIR": "checkpoints",
        "RUN_NAME": "ppo_drone"
    }
    
    rng = jax.random.PRNGKey(42)
    train_jit = jax.jit(make_train(config))
    print("Starting training...")
    final_state, metrics = train_jit(rng)
    print("Training finished.")

