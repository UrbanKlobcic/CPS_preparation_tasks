"""
Task 3.4: Standardized Evaluation Interface

Loads a saved policy checkpoint and evaluates it using the black-box transition function.

Usage:
    python eval_agent.py --checkpoint checkpoints/ppo_drone.ckpt --steps 2000 --out eval.npy
"""
import argparse
import numpy as np
import jax
import jax.numpy as jnp
import acro_step_runtime

from checkpoints import load_checkpoint
from network import ActorCritic
from drone_race_env import DroneRaceEnv, DEFAULT_PARAMS
from plot_density import plot_density

def blackbox_dynamics_adapter(x, u, dt):
    def host_call(x_in, u_in):
        return np.array(acro_step_runtime.step(x_in, u_in), dtype=np.float32)

    return jax.pure_callback(host_call, jax.ShapeDtypeStruct(x.shape, jnp.float32), x, u)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--out", type=str, default="evaluation.npy")
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    params = load_checkpoint(args.checkpoint)

    env = DroneRaceEnv(dynamics_fn=blackbox_dynamics_adapter)

    network = ActorCritic(env.action_size, activation="relu")

    eval_params = DEFAULT_PARAMS._replace(
        max_episode_steps=args.steps,
        initial_gate_id=0,
        noise_pos=0.0,
        noise_vel=0.0,
        noise_ori=0.0,
        noise_rate=0.0
    )

    rng = jax.random.PRNGKey(0)
    rng_reset, rng_loop = jax.random.split(rng)

    obs, state = env.reset(rng_reset, eval_params)

    print(f"Starting evaluation loop for {args.steps} steps...")

    @jax.jit
    def run_eval(rng, init_obs, init_state):
        def scan_step(carry, _):
            rng, obs, state, done = carry

            pi, _ = network.apply(params, obs)
            action = pi.mode()
            action = jnp.clip(action, -1.0, 1.0)

            rng, step_rng = jax.random.split(rng)

            next_obs, next_state, _, new_done, _ = env.step(step_rng, state, action, eval_params)

            done = done | new_done

            carry = (rng, next_obs, next_state, done)
            output = (state.x, action, done)
            return carry, output

        _, (xs, us, done) = jax.lax.scan(
            scan_step,
            (rng, init_obs, init_state, False),
            None,
            length=args.steps
        )
        return xs, us, done

    xs, us, done = run_eval(rng_loop, obs, state)

    xs_np = np.array(xs)
    us_np = np.array(us)
    done_np = np.array(done)

    if np.any(done_np):
        valid_len = np.argmax(done_np) + 1
        print(f"Terminated at step {valid_len-1}.")
    else:
        valid_len = args.steps

    rollout = []
    for i in range(valid_len):
        rollout.append((xs_np[i], us_np[i]))

    print(f"Evaluation finished. Saving rollout to {args.out}...")
    np.save(args.out, np.array(rollout, dtype=object))
    print("Rollout saved.")

    img_buf = plot_density(params)
    heatmap_path = "position_density.png"
    with open(heatmap_path, "wb") as f:
        f.write(img_buf.getvalue())

    print(f"Saved heatmap to {heatmap_path}")

if __name__ == '__main__':
    main()
