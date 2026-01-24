import io
import mlflow
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from network import ActorCritic
from drone_race_env import DroneRaceEnv, DEFAULT_PARAMS, ENVIRONMENT, START_POS, SIM_HZ
from utils import quat_rotate
from rerun_viz import visualize_state_action_sequence


def generate_density_heatmap(trajectory):
    """
    Task 3.3: Log a position density visualization.
    """
    # trajectory is list of (state, action) tuples
    pos_x = [float(step[0][0]) for step in trajectory]
    pos_y = [float(step[0][1]) for step in trajectory]
    
    pos_x_arr = np.array(pos_x)
    pos_y_arr = np.array(pos_y)
    
    gates_np = np.array(ENVIRONMENT)
    gate_x = gates_np[:, 1]
    gate_y = gates_np[:, 2]
    
    # Dynamic margin based on trajectory to ensure we see the drone
    traj_min_x, traj_max_x = pos_x_arr.min(), pos_x_arr.max()
    traj_min_y, traj_max_y = pos_y_arr.min(), pos_y_arr.max()
    
    margin = 5.0
    x_min = min(gate_x.min(), traj_min_x) - margin
    x_max = max(gate_x.max(), traj_max_x) + margin
    y_min = min(gate_y.min(), traj_min_y) - margin
    y_max = max(gate_y.max(), traj_max_y) + margin
    
    bins_x = np.arange(x_min, x_max, 0.1)
    bins_y = np.arange(y_min, y_max, 0.1)

    fig, ax = plt.subplots(figsize=(10, 8))
    
    h = ax.hist2d(pos_x_arr, pos_y_arr, bins=[bins_x, bins_y], cmin=1, cmap='viridis')
    fig.colorbar(h[3], ax=ax, label='Visit Count')
    
    # Draw Gates
    gate_width = DEFAULT_PARAMS.gate_radius * 2.0
    for gate in ENVIRONMENT:
        gate_id = int(gate[0])
        pos = gate[1:4]
        quat = gate[4:8]
        
        local_dir = jnp.array([1.0, 0.0, 0.0])
        world_dir = quat_rotate(quat, local_dir)
        
        cx, cy = float(pos[0]), float(pos[1])
        dx, dy = float(world_dir[0]), float(world_dir[1])
        
        x1 = cx - dx * (gate_width / 2)
        y1 = cy - dy * (gate_width / 2)
        x2 = cx + dx * (gate_width / 2)
        y2 = cy + dy * (gate_width / 2)
        
        ax.plot([x1, x2], [y1, y2], color='red', linewidth=2, solid_capstyle='round')
        ax.text(cx, cy, f"G{gate_id}", color='white', ha='center', va='center')

    # Draw Start
    ax.scatter(float(START_POS[0]), float(START_POS[1]), c='green', s=100, marker='x', label='Start')
    
    # Draw Trajectory Line
    ax.plot(pos_x_arr, pos_y_arr, c='white', alpha=0.3, linewidth=1, label='Trajectory')

    ax.set_title(f"Drone Position Density ({len(trajectory)} steps)")
    ax.set_xlabel("World X (m)")
    ax.set_ylabel("World Y (m)")
    ax.set_aspect('equal')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    plt.close(fig)
    return buf


def evaluate_and_export(train_state, config):
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
    total_reward = 0.0
    
    # Track actions to debug "Straight Line" issue
    action_log = []
    
    while steps < eval_params.max_episode_steps:
        pi, _ = network.apply(train_state.params, obs)
        action = pi.mode()
        action = jnp.clip(action, -1.0, 1.0)
        
        # Store for viz
        trajectory.append((np.array(state.x), np.array(action)))
        action_log.append(np.array(action))
        
        rng, step_rng = jax.random.split(rng)
        obs, state, reward, done, info = env.step(step_rng, state, action, eval_params)
        
        total_reward += reward
        steps += 1
        if done:
            print(f"Eval Episode ended at step {steps}. Info: {info}")
            break
    
    # Action Analysis
    action_arr = np.array(action_log)
    print("\n--- Action Statistics (Eval) ---")
    print(f"Mean Action: {np.mean(action_arr, axis=0)}")
    print(f"Std Action:  {np.std(action_arr, axis=0)}")
    print(f"Max Action:  {np.max(action_arr, axis=0)}")
    print(f"Min Action:  {np.min(action_arr, axis=0)}")
    print("--------------------------------\n")

    print(f"Evaluation finished. Steps: {len(trajectory)}, Total Reward: {total_reward}")
    
    # 1. Save Raw Rollout (.npy)
    npy_path = "evaluation_rollout.npy"
    np.save(npy_path, np.array(trajectory, dtype=object))
    print(f"Saved rollout to {npy_path}")

    mlflow.log_artifact(npy_path)
    
    # 2. Save Rerun Visualization (.rrd)
    rrd_path = "rollout.rrd"
    print(f"Generating Rerun recording to {rrd_path}...")
    try:
        # Convert JAX environment array to numpy for Rerun
        gates_np = np.array(ENVIRONMENT)
        visualize_state_action_sequence(
            sequence=trajectory,
            gates=gates_np,
            recording_path=rrd_path,
            app_id=f"eval_{config.get('RUN_NAME', 'drone')}"
        )
        print("Rerun recording generated.")
        mlflow.log_artifact(rrd_path)
    except Exception as e:
        print(f"Failed to generate Rerun visualization: {e}")

    # 3. Save Density Heatmap (.png)
    img_buf = generate_density_heatmap(trajectory)
    heatmap_path = "position_density.png"
    with open(heatmap_path, "wb") as f:
        f.write(img_buf.getvalue())
    
    print(f"Saved density heatmap to {heatmap_path}")
    mlflow.log_artifact(heatmap_path)
