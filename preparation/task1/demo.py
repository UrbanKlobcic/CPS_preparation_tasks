# task1/demo.py

import jax
import jax.numpy as jnp

from gridworld import GridWorldEnv


DIRECTION_SYMBOLS = {
    0: ">",  # right
    1: "v",  # down
    2: "<",  # left
    3: "^",  # up
}


def render_grid(env: GridWorldEnv, obs, step_idx: int, reward: float, done: bool, info):
    """Print a simple ASCII rendering of the grid."""
    size = env.size

    # Create empty grid filled with dots
    grid = [["." for _ in range(size)] for _ in range(size)]

    ax, ay = map(int, obs["agent"])   # agent position
    tx, ty = map(int, obs["target"])  # target position
    d = int(obs["direction"])         # direction index

    # Place target
    if ax == tx and ay == ty:
        # Agent on target
        grid[ty][tx] = "X"
    else:
        grid[ty][tx] = "T"
        grid[ay][ax] = DIRECTION_SYMBOLS.get(d, "?")

    # Print
    print(f"\n=== Step {step_idx} ===")
    print(f"Reward: {reward:.1f}, Done: {done}, Info: {info}")

    # Top border
    print("+" + "-" * (2 * size - 1) + "+")
    for row in range(size):
        line = "|"
        for col in range(size):
            line += grid[row][col]
            if col < size - 1:
                line += " "
        line += "|"
        print(line)
    # Bottom border
    print("+" + "-" * (2 * size - 1) + "+")


def main():
    env = GridWorldEnv(size=5)

    # RNG setup
    rng = jax.random.PRNGKey(0)
    obs, state = env.reset(rng)

    print("Initial state:")
    render_grid(env, obs, step_idx=0, reward=0.0, done=False, info={"distance": "n/a"})

    # Run a short random rollout
    num_steps = 1000

    for t in range(1, num_steps + 1):
        rng, rng_action = jax.random.split(rng)
        action = jax.random.randint(rng_action, (), 0, env.num_actions, dtype=jnp.int32)

        obs, state, reward, done, info = env.step(rng, state, action)

        print(f"\nAction taken: {int(action)}")
        render_grid(env, obs, step_idx=t, reward=float(reward), done=bool(done), info=info)

        if bool(done):
            print("\nReached the target! 🎉")
            break


if __name__ == "__main__":
    main()
