# File: task3/evaluate.py

import os
import argparse
from functools import partial

import jax
import jax.numpy as jnp

from gridworld import GridWorldEnv
from reinforce import ReinforceAgent, state_to_features

# Action mapping
ACTION_NAMES = ["no-op", "forward", "left", "right"]

# 0 right, 1 down, 2 left, 3 up
DIR_ARROWS = {
    0: ">",
    1: "v",
    2: "<",
    3: "^",
}


def render_ascii(size, agent_pos, target_pos, direction):
    """
    Render the GridWorldEnv in the terminal.

    Coordinates:
        agent_pos = [x, y]
        x increases to the right
        y increases downward
    """
    ax, ay = int(agent_pos[0]), int(agent_pos[1])
    tx, ty = int(target_pos[0]), int(target_pos[1])
    d = int(direction)

    grid = [["." for _ in range(size)] for _ in range(size)]

    # Place target
    grid[ty][tx] = "T"

    # Render agent (with direction)
    agent_char = DIR_ARROWS.get(d, "A")
    if ax == tx and ay == ty:
        agent_char = "X"  # agent on target
    grid[ay][ax] = agent_char

    # Border for readability
    lines = []
    lines.append("+" + "-" * (2 * size - 1) + "+")
    for row in grid:
        lines.append("|" + " ".join(row) + "|")
    lines.append("+" + "-" * (2 * size - 1) + "+")
    return "\n".join(lines)


def _host_print_step(step_idx, action, agent_pos, target_pos, direction, reward, done, distance, size):
    """
    Host-side printing function used by jax.debug.callback.
    Values arrive as numpy scalars/arrays on the host.
    """
    step_idx = int(step_idx)
    action = int(action)
    direction = int(direction)
    reward = float(reward)
    done = bool(done)
    distance = int(distance)

    if step_idx < 0:
        print("\n=== Evaluation episode ===")
        print("Initial state:")
        print(render_ascii(size, agent_pos, target_pos, direction))
        return

    action_name = ACTION_NAMES[action] if 0 <= action < len(ACTION_NAMES) else str(action)

    print(f"\nStep {step_idx + 1}")
    print(f"Action: {action} ({action_name})")
    print(f"Manhattan distance: {distance}")
    print(render_ascii(size, agent_pos, target_pos, direction))

    if done:
        print(f"\nDone! reward={reward} in {step_idx + 1} steps.")


def make_greedy_runner(env: GridWorldEnv, agent: ReinforceAgent):
    """
    Returns a JIT-compiled greedy rollout function that:
      - uses the agent's policy params
      - prints via jax.debug.callback when debug=True
    """
    size = env.size
    num_dirs = agent.num_dirs

    def best_action(params, state):
        x = state_to_features(state, size, num_dirs)
        logits = agent.model.apply(params, x)
        return jnp.argmax(logits, axis=-1).astype(jnp.int32)

    @partial(jax.jit, static_argnames=("max_steps", "debug"))
    def run_greedy(params, rng, max_steps=100, debug=False):
        rng, reset_rng = jax.random.split(rng)
        obs, state = env.reset(reset_rng)

        if debug:
            agent_pos, target_pos, direction = state
            distance0 = jnp.sum(jnp.abs(agent_pos - target_pos)).astype(jnp.int32)

            jax.debug.callback(
                _host_print_step,
                jnp.int32(-1),
                jnp.int32(-1),
                agent_pos,
                target_pos,
                direction,
                jnp.float32(0.0),
                jnp.bool_(False),
                distance0,
                size,
            )

        # Carry: (state, done, rng, steps_taken)
        def scan_step(carry, t):
            state, done, rng, steps_taken = carry
            active = jnp.logical_not(done)

            rng, step_rng = jax.random.split(rng)
            action = best_action(params, state)

            obs2, next_state, reward, step_done, info = env.step(step_rng, state, action)
            agent_pos2, target_pos2, direction2 = next_state
            distance = jnp.sum(jnp.abs(agent_pos2 - target_pos2)).astype(jnp.int32)

            def select_state(old_s, new_s):
                return jax.tree_util.tree_map(
                    lambda o, n: jnp.where(active, n, o),
                    old_s,
                    new_s,
                )

            state2 = select_state(state, next_state)
            done2 = jnp.where(active, step_done, done)

            # Count steps until first done
            steps_taken2 = steps_taken + active.astype(jnp.int32)

            # Debug printing
            if debug:
                def _do_cb(_):
                    jax.debug.callback(
                        _host_print_step,
                        t,
                        action,
                        agent_pos2,
                        target_pos2,
                        direction2,
                        reward,
                        step_done,
                        distance,
                        size,
                    )
                    return 0

                def _no_cb(_):
                    return 0

                jax.lax.cond(active, _do_cb, _no_cb, operand=0)

            return (state2, done2, rng, steps_taken2), None

        (final_state, final_done, rng, steps_taken), _ = jax.lax.scan(
            scan_step,
            (state, jnp.bool_(False), rng, jnp.int32(0)),
            xs=jnp.arange(max_steps, dtype=jnp.int32),
        )

        return final_state, final_done, steps_taken, rng

    return run_greedy


def run_episode(agent: ReinforceAgent, max_steps=100, seed=123, debug=False):
    """
    Thin Python wrapper around the JIT greedy runner.
    """
    env = agent.env
    runner = make_greedy_runner(env, agent)

    rng = jax.random.PRNGKey(seed)

    final_state, done, steps_taken, rng = runner(
        agent.state.params, rng, max_steps=max_steps, debug=debug
    )

    # If debug is off, only print a summary
    if not debug:
        done_py = bool(done)
        steps_py = int(steps_taken)

        agent_pos, target_pos, direction = final_state
        distance = int(jnp.sum(jnp.abs(agent_pos - target_pos)))

        print("\n=== Evaluation episode (greedy) ===")
        print(f"Finished. done={done_py}, steps_taken={steps_py}, final_distance={distance}")
        print(render_ascii(env.size, agent_pos, target_pos, direction))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=5)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--debug", action="store_true",
                        help="Enable step-by-step ASCII printing via jax.debug.callback")
    parser.add_argument("--policy_path", type=str, default="policy_params.msgpack")
    args = parser.parse_args()

    env = GridWorldEnv(size=args.size)

    agent = ReinforceAgent(
        env,
        num_episodes=1,
        max_steps=args.max_steps,
        hidden_size=args.hidden_size,
        seed=args.seed,
    )

    if not os.path.exists(args.policy_path):
        raise FileNotFoundError(
            f"Could not find {args.policy_path}. Please run reinforce.py first "
            f"or provide --policy_path."
        )

    agent.load_params(args.policy_path)
    print("Loaded policy params from:", args.policy_path)

    run_episode(agent, max_steps=args.max_steps, seed=args.seed, debug=args.debug)
