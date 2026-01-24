"""
Task 3.4: Standardized Evaluation Interface

Loads a saved policy checkpoint and evaluates it using the black-box transition function.

Usage:
    python eval_agent.py --checkpoint checkpoints/ppo_drone.ckpt --steps 2000 --out eval.npy
"""
import argparse
import numpy as np
import jax

import acro_step_runtime
from checkpoints import load_checkpoint
from network import ActorCritic
from drone_race_env import DroneRaceEnv, DEFAULT_PARAMS

def blackbox_dynamics_adapter(x, u, dt):
    return acro_step_runtime.step(x, u)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to policy checkpoint")
    parser.add_argument("--steps", type=int, default=2000, help="Number of steps to run")
    parser.add_argument("--out", type=str, default="evaluation.npy", help="Output .npy file path")
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    params = load_checkpoint(args.checkpoint)
    
    # 1. Instantiate Env with Black-Box Dynamics
    # We pass the adapter which calls acro_step_runtime.step
    env = DroneRaceEnv(dynamics_fn=blackbox_dynamics_adapter)
    
    # 2. Init Network (must match training config)
    network = ActorCritic(env.action_size, activation="tanh")
    
    # 3. Setup Evaluation Params (No Noise, Long Horizon)
    eval_params = DEFAULT_PARAMS._replace(
        max_episode_steps=args.steps + 100, # Ensure env doesn't timeout early
        initial_gate_id=0,
        noise_pos=0.0, 
        noise_vel=0.0, 
        noise_ori=0.0, 
        noise_rate=0.0
    )
    
    # 4. Reset
    rng = jax.random.PRNGKey(0)
    rng_reset, rng_loop = jax.random.split(rng)
    
    obs, state = env.reset(rng_reset, eval_params)
    
    rollout = [] # List of (x, u) tuples for Task 2
    
    @jax.jit
    def policy_step(params, obs):
        pi, _ = network.apply(params, obs)
        return pi.mode()
    
    @jax.jit
    def env_step(rng, state, action):
        return env.step(rng, state, action, eval_params)

    print(f"Starting evaluation loop for {args.steps} steps...")
    
    for t in range(args.steps):
        # Select Action
        u = policy_step(params, obs)
        
        # Store State/Action (convert to numpy for serialization)
        x_np = np.array(state.x)
        u_np = np.array(u)
        rollout.append((x_np, u_np))
        
        # Step Env
        rng_loop, step_key = jax.random.split(rng_loop)
        obs, state, reward, done, info = env_step(step_key, state, u)
        
        if done:
            print(f"Terminated at step {t}.")
            if info["out_of_bounds"]:
                print("Reason: Out of Bounds / Crash")
            if state.x[20] < 22.0:
                print("Reason: Battery Depleted")
            break
            
        if (t+1) % 500 == 0:
            print(f"Step {t+1}/{args.steps} | Gates Passed: {state.gates_passed}")

    print(f"Evaluation finished. Saving rollout to {args.out}...")
    np.save(args.out, np.array(rollout, dtype=object))
    print("Done.")

if __name__ == '__main__':
    main()
