import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
import mlflow
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple
from flax.training.train_state import TrainState
import distrax
from wrappers import LogWrapper, FlattenObservationWrapper, VecEnv, AutoResetWrapper
from drone_race_env import DroneRaceEnv, DEFAULT_PARAMS
import checkpoints
import eval
from network import ActorCritic

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray

def make_train(config):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    
    # 1. Init Env
    env = DroneRaceEnv()
    env_params = DEFAULT_PARAMS
    
    # 2. Wrap (Order: Flatten -> Log -> AutoReset -> VecEnv)
    env = FlattenObservationWrapper(env)
    env = LogWrapper(env)
    env = AutoResetWrapper(env) # Handles resetting logic internally
    env = VecEnv(env)           # Handles vmap internally

    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac

    def train(rng):
        # INIT NETWORK
        network = ActorCritic(
            env.action_space(env_params).shape[0], activation=config["ACTIVATION"]
        )
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(env.observation_space(env_params).shape)
        network_params = network.init(_rng, init_x)
        
        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(learning_rate=linear_schedule, eps=1e-5),
        )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = env.reset(reset_rng, env_params)

        # TRAIN LOOP
        def _update_step(runner_state, update_i):
            
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                pi, value = network.apply(train_state.params, last_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                
                # Baseline logic: env.step handles vmap and reset
                obsv, env_state, reward, done, info = env.step(
                    rng_step, env_state, action, env_params
                )
                
                transition = Transition(
                    done, action, value, reward, log_prob, last_obs, info
                )
                runner_state = (train_state, env_state, obsv, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )
            
            metric = traj_batch.info

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, rng = runner_state
            _, last_val = network.apply(train_state.params, last_obs)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        pi, value = network.apply(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree.map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree.map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree.map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            rng = update_state[-1]
            
            loss_info = jax.tree.map(lambda x: x.mean(), loss_info)

            # Debug Callback
            if config.get("DEBUG"):
                def callback(info, loss_info, update_iter):
                    _, (v_loss, a_loss, ent) = loss_info
                    mask = info["returned_episode"]
                    
                    avg_ret = 0.0
                    avg_len = 0.0
                    avg_gates = 0.0
                    crash_rate = 0.0
                    
                    if np.any(mask):
                        avg_ret = float(np.sum(info["returned_episode_returns"] * mask) / np.sum(mask))
                        avg_len = float(np.sum(info["returned_episode_lengths"] * mask) / np.sum(mask))
                        avg_gates = float(np.sum(info["gates_passed"] * mask) / np.sum(mask))
                        crash_rate = float(np.sum(info["out_of_bounds"] * mask) / np.sum(mask))

                    global_step = (update_iter + 1) * config["NUM_STEPS"] * config["NUM_ENVS"]
                    
                    print(
                        f"Step {global_step:<9} | "
                        f"Ret: {avg_ret:<7.2f} | "
                        f"Len: {avg_len:<5.1f} | "
                        f"Gates: {avg_gates:<5.2f} | "
                        f"OOB: {crash_rate:<4.2f} | "
                        f"ValLoss: {float(v_loss):<7.4f}"
                    )

                    if mlflow.active_run():
                        mlflow.log_metrics({
                            "mean_return": avg_ret,
                            "mean_length": avg_len,
                            "gates_passed": avg_gates,
                            "crash_rate": crash_rate,
                            "value_loss": float(v_loss),
                            "entropy": float(ent),
                        }, step=int(global_step))

                jax.debug.callback(callback, metric, loss_info, update_i)

            runner_state = (train_state, env_state, last_obs, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, _rng)
        
        update_indices = jnp.arange(config["NUM_UPDATES"])
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, update_indices
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train

if __name__ == "__main__":
    config = {
        "LR": 3e-4,
        "NUM_ENVS": 256,
        "NUM_STEPS": 256,
        "TOTAL_TIMESTEPS": 5e6,
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 4,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ACTIVATION": "tanh",
        "ANNEAL_LR": True,
        "RUN_NAME": "ppo_drone_race",
        "CKPT_DIR": "checkpoints",
        "DEBUG": True,
    }

    mlflow.set_experiment("DroneRacing_PPO")
    with mlflow.start_run(run_name=config["RUN_NAME"]):
        mlflow.log_params(config)
        rng = jax.random.PRNGKey(42)
        train_jit = jax.jit(make_train(config))
        print("Starting training...")
        out = train_jit(rng)
        print("Training finished.")
        checkpoints.save_checkpoint(out["runner_state"][0], config)
        eval.evaluate_and_export(out["runner_state"][0], config)

