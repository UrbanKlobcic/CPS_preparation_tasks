"""
Smaller eval script that runs a rollout with the fitted model, not the blackbox.
Logs the trajectory and density heatmap to MLflow.
"""
import mlflow
import jax
import jax.numpy as jnp
import numpy as np

from network import ActorCritic
from drone_race_env import DroneRaceEnv, DEFAULT_PARAMS, ENVIRONMENT, SIM_HZ
from plot_density import plot_density
from rerun_viz import visualize_state_action_sequence

def eval_and_log_artifacts(train_state, config):
    print("Running evaluation rollout...")
    env = DroneRaceEnv()
    
    # zero noise for evaluation + start at gate 0
    eval_params = DEFAULT_PARAMS._replace(
        max_episode_steps=SIM_HZ * 60,
        initial_gate_id=0,
        noise_pos=0.0,
        noise_vel=0.0,
        noise_ori=0.0,
        noise_rate=0.0
    )
    
    rng = jax.random.PRNGKey(0)
    obs, state = env.reset(rng, eval_params)
    
    trajectory = [] 
    
    network = ActorCritic(env.action_size, activation=config["ACTIVATION"])
    
    steps = 0
    passed_gates = 0
    total_reward = 0.0
    done = False
    action_log = []
    
    while steps < eval_params.max_episode_steps and not done:
        pi, _ = network.apply(train_state.params, obs)
        action = pi.mode()
        action = jnp.clip(action, -1.0, 1.0)
        
        trajectory.append((np.array(state.x), np.array(action)))
        action_log.append(np.array(action))
        
        rng, step_rng = jax.random.split(rng)
        obs, state, reward, done, info = env.step(step_rng, state, action, eval_params)
        
        passed_gates += info["gates_passed"]
        total_reward += reward
        steps += 1
    
    action_arr = np.array(action_log)
    print("\n--- Action Statistics (Eval) ---")
    print(f"Mean Action: {np.mean(action_arr, axis=0)}")
    print(f"Std Action:  {np.std(action_arr, axis=0)}")
    print(f"Max Action:  {np.max(action_arr, axis=0)}")
    print(f"Min Action:  {np.min(action_arr, axis=0)}")
    print("--------------------------------\n")
    
    print(f"Evaluation finished. Steps: {len(trajectory)}, Gates Passed: {passed_gates}, Total Reward: {total_reward}")
    
    npy_path = "evaluation_rollout.npy"
    np.save(npy_path, np.array(trajectory, dtype=object))
    print(f"Saved rollout to {npy_path}")
    
    if mlflow.active_run():
        mlflow.log_artifact(npy_path)
    
    rrd_path = "rollout.rrd"
    print(f"Generating Rerun recording to {rrd_path}...")
    
    gates_np = np.array(ENVIRONMENT)
    visualize_state_action_sequence(
        sequence=trajectory,
        gates=gates_np,
        recording_path=rrd_path,
        app_id=f"eval_{config['RUN_NAME']}"
    )
    
    print("Rerun recording generated.")
    if mlflow.active_run():
        mlflow.log_artifact(rrd_path)
    
    img_buf = plot_density(train_state.params)
    heatmap_path = "position_density.png"
    with open(heatmap_path, "wb") as f:
        f.write(img_buf.getvalue())
    
    print(f"Saved density heatmap to {heatmap_path}")
    
    if mlflow.active_run():
        mlflow.log_artifact(heatmap_path)

