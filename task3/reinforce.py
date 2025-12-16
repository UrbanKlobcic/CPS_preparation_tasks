# File: task3/reinforce.py

from typing import Any, Optional

import jax
import jax.numpy as jnp
import flax.linen as nn
import flax.serialization as serialization
from flax import struct
import optax

from gridworld import GridWorldEnv


class PolicyNet(nn.Module):
    num_actions: int
    hidden_size: int = 64

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.hidden_size)(x)  # First hidden layer
        x = nn.tanh(x)                     # Activation
        x = nn.Dense(self.hidden_size)(x)  # Second hidden layer
        x = nn.tanh(x)                     # Activation
        return nn.Dense(self.num_actions)(x)


def state_to_features(state, size: int, num_dirs: int) -> jnp.ndarray:
    """
    Convert environment state to feature vector for policy network.
    """
    agent_pos, target_pos, direction = state

    denom = jnp.maximum(size - 1, 1)
    agent = agent_pos.astype(jnp.float32) / denom
    target = target_pos.astype(jnp.float32) / denom
    dir_oh = jax.nn.one_hot(direction, num_dirs, dtype=jnp.float32)

    return jnp.concatenate([agent, target, dir_oh], axis=-1)


def logprob_from_logits(logits, action):
    """
    Compute log-probability of selected action given logits.
    """
    return jax.nn.log_softmax(logits)[action]


@struct.dataclass
class ReinforceTrainState:
    params: Any
    opt_state: Any
    rng: jax.Array


class ReinforceAgent:
    """
    Vanilla REINFORCE in JAX.

    - Roll out one episode (max_steps cap).
    - Compute discounted returns.
    - Update policy with classic REINFORCE loss.
    """

    def __init__(
        self,
        env: GridWorldEnv,
        learning_rate=3e-4,
        discount_factor=0.95,
        num_episodes=2000,
        max_steps=50,
        hidden_size=64,
        seed=0,
    ):
        self.env = env
        self.size = int(env.size)
        self.num_dirs = int(env.num_dirs)
        self.num_actions = int(env.num_actions)

        self.gamma = jnp.asarray(discount_factor, jnp.float32)
        self.num_episodes = int(num_episodes)
        self.max_steps = int(max_steps)

        self.model = PolicyNet(self.num_actions, int(hidden_size))
        self.optimizer = optax.adam(learning_rate)

        rng = jax.random.PRNGKey(seed)
        rng, init_rng, reset_rng = jax.random.split(rng, 3)
        _, init_state = self.env.reset(reset_rng)
        dummy_x = state_to_features(init_state, self.size, self.num_dirs)

        params = self.model.init(init_rng, dummy_x)
        opt_state = self.optimizer.init(params)

        self.state = ReinforceTrainState(params=params, opt_state=opt_state, rng=rng)

        self._train_n_episodes_jit = jax.jit(
            self._train_n_episodes,
            static_argnames=("n_episodes", "max_steps"),
        )

    # Serialization helpers
    def save_params(self, path: str):
        """
        Save policy parameters using Flax serialization.
        """
        with open(path, "wb") as f:
            f.write(serialization.to_bytes(self.state.params))

    def load_params(self, path: str):
        """
        Load policy parameters using Flax serialization.
        The agent must be constructed with the same architecture
        (hidden_size, num_actions, etc.) so the parameter structure matches.
        """
        with open(path, "rb") as f:
            encoded_bytes = f.read()

        params = serialization.from_bytes(self.state.params, encoded_bytes)

        self.state = ReinforceTrainState(
            params=params,
            opt_state=self.state.opt_state,
            rng=self.state.rng,
        )

    # Episode rollout (fixed length + mask)
    def _rollout(self, params, rng, init_state, max_steps: int):
        """
        Returns:
          logps:   (T,)
          rewards: (T,)
          actives: (T,) 1.0 until done, then 0.0
          rng_out
        """

        def step_fn(carry, _):
            state, done, key = carry
            key, key_act, key_step = jax.random.split(key, 3)

            active = jnp.logical_not(done)

            x = state_to_features(state, self.size, self.num_dirs)
            logits = self.model.apply(params, x)

            action = jax.random.categorical(key_act, logits).astype(jnp.int32)
            logp = logprob_from_logits(logits, action)

            _, next_state, reward, step_done, _ = self.env.step(key_step, state, action)

            # mask after done
            reward = jnp.where(active, reward, 0.0)
            logp = jnp.where(active, logp, 0.0)

            done2 = jnp.logical_or(done, jnp.where(active, step_done, False))

            state2 = jax.tree_util.tree_map(
                lambda s, ns: jnp.where(active, ns, s),
                state,
                next_state,
            )

            return (state2, done2, key), (logp, reward, active.astype(jnp.float32))

        (_, _, rng_out), (logps, rewards, actives) = jax.lax.scan(
            step_fn,
            (init_state, jnp.bool_(False), rng),
            xs=None,
            length=max_steps,
        )

        return logps, rewards, actives, rng_out

    # Discounted returns
    def _returns(self, rewards):
        def step(g_t1, r_t):
            g_t = r_t + self.gamma * g_t1
            return g_t, g_t # (carry, return value)

        return jax.lax.scan(step, 0.0, rewards, reverse=True)[1]

    # One episode update
    def _train_one_episode(self, train_state: ReinforceTrainState, max_steps: int):
        params, opt_state, rng = (
            train_state.params,
            train_state.opt_state,
            train_state.rng,
        )

        rng, rng_reset, rng_ep = jax.random.split(rng, 3)
        _, init_state = self.env.reset(rng_reset)

        def loss_fn(p):
            logps, rewards, actives, rng_out = self._rollout(
                p, rng_ep, init_state, max_steps
            )
            returns = self._returns(rewards)

            # Classic REINFORCE loss (mask by actives)
            loss = -jnp.sum(logps * jax.lax.stop_gradient(returns) * actives)
            ep_return = jnp.sum(rewards)

            return loss, (ep_return, rng_out)

        (loss, (ep_return, rng_out)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params
        )

        updates, opt_state = self.optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        new_state = ReinforceTrainState(params=params, opt_state=opt_state, rng=rng_out)
        metrics = {"episode_return": ep_return, "loss": loss}
        return new_state, metrics

    # Train for n episodes
    def _train_n_episodes(
        self, train_state: ReinforceTrainState, n_episodes: int, max_steps: int
    ):
        def body(carry, _):
            carry, metrics = self._train_one_episode(carry, max_steps)
            return carry, metrics

        train_state, metrics = jax.lax.scan(
            body, train_state, xs=None, length=n_episodes
        )
        return train_state, metrics

    # Public API
    def train(
        self, num_episodes: Optional[int] = None, max_steps: Optional[int] = None
    ):
        n_eps = self.num_episodes if num_episodes is None else int(num_episodes)
        m_steps = self.max_steps if max_steps is None else int(max_steps)

        self.state, metrics = self._train_n_episodes_jit(
            self.state, n_episodes=n_eps, max_steps=m_steps
        )
        return metrics

    def get_greedy_action(self, env_state, params=None):
        if params is None:
            params = self.state.params
        x = state_to_features(env_state, self.size, self.num_dirs)
        logits = self.model.apply(params, x)
        return jnp.argmax(logits).astype(jnp.int32)


if __name__ == "__main__":
    env = GridWorldEnv(size=5)

    agent = ReinforceAgent(
        env,
        learning_rate=3e-4,
        discount_factor=0.95,
        num_episodes=20_000,
        max_steps=50,
        hidden_size=64,
        seed=0,
    )

    print("REINFORCE agent initialized.")
    metrics = agent.train()

    returns = metrics["episode_return"]
    print("Training finished.")
    print("Last episode return:", float(returns[-1]))
    print("Mean return (last 100):", float(jnp.mean(returns[-100:])))

    # Save policy params
    path = "../learned_policies/policy_params.msgpack"
    agent.save_params(path)
    print(f"Saved policy params to {path}")
