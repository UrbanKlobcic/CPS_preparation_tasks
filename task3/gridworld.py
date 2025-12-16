# task1/gridworld.py

import jax
import jax.numpy as jnp
from typing import NamedTuple, Dict, Tuple, Any


class GridWorldState(NamedTuple):
    agent_pos: jnp.ndarray  # shape (2,)
    target_pos: jnp.ndarray  # shape (2,)
    direction: jnp.ndarray  # scalar int


class GridWorldEnv:
    """
    JAX GridWorld environment.

    - Actions (int):
        0: no-op
        1: move forward (in current direction)
        2: rotate left
        3: rotate right

    - Observation (dict of JAX arrays):
        {
            "agent": (2,) int array in [0, size-1]
            "target": (2,) int array in [0, size-1]
            "direction": scalar int in [0, 3]
        }
    """

    def __init__(self, size: int = 5):
        self.size = int(size)
        self.num_dirs = 4
        self.num_actions = 4

        # dir index: 0 right, 1 down, 2 left, 3 up
        self.direction_to_move = jnp.array(
            [
                [1, 0],  # right
                [0, 1],  # down
                [-1, 0],  # left
                [0, -1],  # up
            ],
            dtype=jnp.int32,
        )

    def reset(self, rng: jax.Array) -> Tuple[Dict[str, jax.Array], GridWorldState]:
        rng_agent, rng_target, rng_dir = jax.random.split(rng, 3)

        agent_pos = jax.random.randint(
            rng_agent, (2,), 0, self.size, dtype=jnp.int32
        )  # TODO: initialize agent position randomly
        target_pos = jax.random.randint(
            rng_target, (2,), 0, self.size, dtype=jnp.int32
        )  # TODO: initialize target position randomly

        # Ensure agent and target are not identical.
        same = jnp.all(
            agent_pos == target_pos
        )  # TODO: check if agent_pos and target_pos are the same position

        # If same, shift target by +1 (wrap) on both coordinates.
        target_pos = jax.lax.cond(
            same,
            lambda _: (target_pos + 1) % self.size,
            lambda _: target_pos,
            operand=None,
        )  # TODO: use jax.lax.cond to conditionally update target_pos

        direction = jax.random.randint(
            rng_dir, (), 0, self.num_dirs, dtype=jnp.int32
        )  # TODO: initialize direction randomly

        state = GridWorldState(
            agent_pos=agent_pos, target_pos=target_pos, direction=direction
        )
        obs = self._get_obs(state)
        return obs, state

    def step(
        self,
        rng: jax.Array,
        state: GridWorldState,
        action: jax.Array,
    ) -> Tuple[
        Dict[str, jax.Array], GridWorldState, jax.Array, jax.Array, Dict[str, Any]
    ]:
        agent_pos, target_pos, direction = state

        def move_forward(pos: jax.Array, dir_idx: jax.Array) -> jax.Array:
            # TODO: compute new position by moving in the current direction (hint: self.direction_to_move)
            step_vec = self.direction_to_move[dir_idx]
            new = pos + step_vec
            # Clip to grid
            new_pos = jnp.clip(new, 0, self.size - 1)

            return new_pos

        # Action == 0: no-op
        new_pos = agent_pos
        new_direction = direction

        # Action == 1: move forward
        new_pos = jax.lax.cond(
            action == 1,
            lambda _: move_forward(agent_pos, direction),
            lambda _: new_pos,
            operand=None,
        )  # TODO: use jax.lax.cond to conditionally update new_pos using move_forward

        # Action == 2: turn left
        new_direction = jax.lax.cond(
            action == 2,
            lambda _: (direction - 1) % self.num_dirs,
            lambda _: new_direction,
            operand=None,
        )  # TODO: use jax.lax.cond to conditionally update new_direction for left turn

        # Action == 3: turn right
        new_direction = new_direction = jax.lax.cond(
            action == 3,
            lambda _: (direction + 1) % self.num_dirs,
            lambda _: new_direction,
            operand=None,
        )  # TODO: use jax.lax.cond to conditionally update new_direction for right turn

        done = jnp.all(
            new_pos == target_pos
        )  # TODO: check if new_pos matches target_pos
        reward = jnp.where(
            done, 1.0, 0.0
        )  # TODO: assign reward of 1.0 if done else 0.0

        new_state = GridWorldState(
            agent_pos=new_pos, target_pos=target_pos, direction=new_direction
        )
        obs = self._get_obs(new_state)

        info = {
            "distance": jnp.sum(
                jnp.abs(new_pos - target_pos)
            ),  # TODO: compute Manhattan distance between new_pos and target_pos
        }
        return obs, new_state, reward, done, info

    def _get_obs(self, state: GridWorldState) -> Dict[str, jax.Array]:
        return {
            "agent": state.agent_pos,
            "target": state.target_pos,
            "direction": state.direction,
        }


if __name__ == "__main__":
    # Test of the environment by applying random actions
    env = GridWorldEnv(size=5)
    rng = jax.random.PRNGKey(0)

    obs, state = env.reset(rng)

    # JIT the step method
    step_jit = jax.jit(env.step)

    for _ in range(1000):
        rng, rng_action = jax.random.split(rng)
        action = jax.random.randint(rng_action, (), 0, env.num_actions, dtype=jnp.int32)

        obs, state, reward, done, info = step_jit(rng, state, action)
        print(
            f"Action: {int(action)}, Obs: {obs}, Reward: {float(reward)}, Done: {bool(done)}, Info: {info}"
        )

        if bool(done):
            break
