import os
import jax
import jax.numpy as jnp
import optax
from flax import serialization
from flax.training.train_state import TrainState
from drone_race_env import DroneRaceEnv
from network import ActorCritic


def load_checkpoint(path):
    print(f"Loading checkpoint from {path}...")
    with open(path, "rb") as f:
        ckpt_bytes = f.read()
    
    env = DroneRaceEnv()
    network = ActorCritic(env.action_size)
    init_x = jnp.zeros((1, env.obs_size))
    
    template_params = network.init(jax.random.PRNGKey(0), init_x)
      
    max_grad_norm = 0.5
    
    def dummy_schedule(_):
        return 3e-4
    
    tx = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(dummy_schedule, eps=1e-5),
    )

    template_state = TrainState.create(
        apply_fn=network.apply, 
        params=template_params, 
        tx=tx
    )

    loaded_state = serialization.from_bytes(template_state, ckpt_bytes)
    
    return loaded_state.params


def save_checkpoint(state: TrainState, config: dict):
    save_dir = config["CKPT_DIR"]
    os.makedirs(save_dir, exist_ok=True)
    
    file_path = os.path.join(save_dir, f"{config['RUN_NAME']}.ckpt")
    with open(file_path, "wb") as f:
        f.write(serialization.to_bytes(state))
    
    print(f"Saved checkpoint to {file_path}")
    return file_path

