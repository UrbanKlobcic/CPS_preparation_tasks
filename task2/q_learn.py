# File: task1/q_learn.py

import os
import time
from typing import Tuple, Any

import jax
import jax.numpy as jnp
import numpy as np  # use numpy ONLY for save/load of q table

from gridworld import GridWorldEnv


class QAgent:
    """
    Tabular Q-learning agent implemented in JAX.

    Env state:
        (agent_pos, target_pos, direction)
        agent_pos: jnp.int32[2]
        target_pos: jnp.int32[2]
        direction: jnp.int32 scalar

    Tabular index corresponds to:
        (x_agent, y_agent, d_agent, x_target, y_target)
    """

    def __init__(
            self,
            env: GridWorldEnv,
            learning_rate=0.1,
            discount_factor=0.95,
            epsilon=0.2,
            train_steps=1_000_000,
            seed=0,
    ):
        self.env = env
        self.learning_rate = jnp.asarray(learning_rate, dtype=jnp.float32)
        self.discount_factor = jnp.asarray(discount_factor, dtype=jnp.float32)
        self.epsilon = jnp.asarray(epsilon, dtype=jnp.float32)
        self.train_steps = int(train_steps)

        self.size = int(env.size)
        self.num_dirs = int(env.num_dirs)
        self.num_actions = int(env.num_actions)

        self.num_states = env.size ** 2 * 4 * env.size**2  # TODO: compute the total number of states

        self.rng = jax.random.PRNGKey(seed)

        # Q-table initialization
        # self.q_table = jnp.zeros((self.num_states, self.num_actions),
        #                          dtype=jnp.float32)  # (Optional TODO: see hint in the assignment)
        self.q_table = jax.random.uniform(
            self.rng,
            (self.num_states, self.num_actions),
            minval=0.001,
            maxval=0.1,
            dtype=jnp.float32
        )

        # JIT-compiled training core
        start_time = time.perf_counter()
        self._train_steps_jit = jax.jit(
            self._train_steps,
            static_argnames=("total_steps",),
        )
        end_time = time.perf_counter()
        print("JIT compilation time:", end_time - start_time, "seconds")

    # State flattening
    def state_to_index(self, state) -> jnp.ndarray:
        """
        Compute unique index for (x_a, y_a, d, x_t, y_t).

        Layout order:
            x_a, then y_a, then d, then x_t, then y_t.

        Returns: jnp.int32 scalar index
        """
        agent_pos, target_pos, direction = state

        S = self.size
        D = self.num_dirs

        x_a = agent_pos[0]
        y_a = agent_pos[1]
        x_t = target_pos[0]
        y_t = target_pos[1]
        d = direction

        # TODO: compute the correct index from the state components
        idx = x_a * (S * D * S * S) + y_a * (D * S * S) + d * (S * S) + x_t * S + y_t
        return idx.astype(jnp.int32)

    # Policy helpers
    def _best_action(self, q_table, state) -> jnp.ndarray:
        """
        Compute the best action for the given state according to the Q-table, e.g. by argmax.

        Returns: jnp.int32 scalar action
        """
        idx = self.state_to_index(state)
        # TODO: return the best action for the given state
        return jnp.argmax(q_table[idx], axis=-1).astype(jnp.int32)

    def _epsilon_greedy(self, q_table, state, rng) -> jnp.ndarray:
        """
        Compute epsilon-greedy action for the given state.

        Either selects the best action (with probability 1 - epsilon)
        or a random action (with probability epsilon).

        Returns: jnp.int32 scalar action
        """
        rng_eps, rng_act = jax.random.split(rng, 2)

        # TODO: implement epsilon-greedy action selection (hint: compute both greedy and random actions and then use jax.lax.select)
        prob = jax.random.uniform(rng_eps)
        greedy_action = self._best_action(q_table, state)
        random_action = jax.random.randint(rng_act, (), 0, self.num_actions, dtype=jnp.int32)
        return jax.lax.select(prob < self.epsilon, random_action, greedy_action)
        return None

    # One-step Q-learning update
    def _q_update(self, q_table, state, action, next_state, reward, done):
        """
        Perform one Q-learning update for the given transition.

        Returns: updated Q-table
        """

        idx = self.state_to_index(state)
        next_idx = self.state_to_index(next_state)

        # TODO: get the current Q-value e.g. Q(s, a)
        current_q = q_table[idx, action]
        # TODO: get the best next Q-value e.g. max_a' Q(s', a')
        best_next_q = jnp.max(q_table[next_idx])

        # Terminal handling
        # TODO: compute the TD target e.g. r + gamma * max_a' Q(s', a') * (1 - done)
        td_target = reward + self.discount_factor * best_next_q * (1.0 - done)
        # TODO: compute the new Q-value e.g. Q(s, a) + alpha * (td_target - Q(s, a))
        new_q = current_q + self.learning_rate * (td_target - current_q)

        q_table = q_table.at[idx, action].set(new_q)
        return q_table

    # JITted training loop over environment steps (has been fully implemented, no TODOs here)
    def _train_steps(
            self,
            q_table: jnp.ndarray,
            rng: jax.random.PRNGKey,
            state: Any,
            total_steps: int,
    ) -> Tuple[jnp.ndarray, jax.random.PRNGKey, Any]:
        """
        Run the environment for 'total_steps' steps.
        If done, reset and continue.
        """

        def step_fn(carry, _):
            q_tab, key, st = carry

            # Split RNG for: action choice, env step, potential reset
            key, key_act, key_step, key_reset = jax.random.split(key, 4)

            # Get action using epsilon-greedy policy
            action = self._epsilon_greedy(q_tab, st, key_act)

            # Take env step using the selected action
            _, next_state, reward, done, _ = self.env.step(key_step, st, action)

            # Update Q-table using the Q-learning update rule
            q_tab = self._q_update(q_tab, st, action, next_state, reward, done)

            # Pre-sample a reset state, select it only if done
            _, reset_state = self.env.reset(key_reset)

            # Select between reset_state and next_state for each leaf
            new_state = jax.tree_util.tree_map(
                lambda rs, ns: jnp.where(done, rs, ns),
                reset_state,
                next_state,
            )

            return (q_tab, key, new_state), None

        (q_table, rng, state), _ = jax.lax.scan(
            step_fn,
            (q_table, rng, state),
            xs=None,
            length=total_steps,
        )
        return q_table, rng, state

    # Public API
    def train(self, total_steps: int | None = None):
        """
        JIT-accelerated training. Runs for 'total_steps' env steps.
        """
        steps = self.train_steps if total_steps is None else int(total_steps)

        self.rng, reset_rng = jax.random.split(self.rng, 2)
        _, state = self.env.reset(reset_rng)

        q_table, rng, _state = self._train_steps_jit(self.q_table, self.rng, state, steps)

        self.q_table = q_table
        self.rng = rng

    # Save/load Q-table using numpy and conversion to/from jax arrays
    def save_q_table(self, file_path):
        np.save(file_path, np.array(self.q_table))

    def load_q_table(self, file_path):
        arr = np.load(file_path)
        self.q_table = jnp.asarray(arr, dtype=jnp.float32)


if __name__ == "__main__":
    env = GridWorldEnv(size=5)

    agent = QAgent(
        env,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=0.4,
        train_steps=1_000_000,  # reduce for quicker testing and see hint in the assignment
        seed=0,
    )

    print("QAgent initialized.")
    print("Q-table shape:", agent.q_table.shape)

    q_path = "q_table.npy"

    if os.path.exists(q_path):
        print("Warning: Q-table file exists and will be overwritten.")
        input("Press Enter to continue...")
        os.remove(q_path)

    start_time = time.perf_counter()
    agent.train(10_000_000)  # runs for agent.train_steps
    end_time = time.perf_counter()
    agent.save_q_table(q_path)
    print("QAgent trained and saved.", "Time taken (s):", (end_time - start_time))

    print("Done.")
