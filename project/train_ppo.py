"""
Train a PPO agent on the drone race env.

Based on purejaxrl PPO implementation
https://github.com/luchris429/purejaxrl/blob/main/purejaxrl/ppo_continuous_action.py

Usage:
    python train_ppo.py [--resume checkpoints/ppo_drone.ckpt]
"""
import argparse
import os
import jax
import jax.numpy as jnp
import numpy as np
import optax
import mlflow
from typing import NamedTuple
from flax.training.train_state import TrainState
from wrappers import LogWrapper, FlattenObservationWrapper, VecEnv, AutoResetWrapper
from drone_race_env import DroneRaceEnv, DEFAULT_PARAMS
from checkpoints import load_checkpoint, save_checkpoint
from eval_and_log import eval_and_log_artifacts
from network import ActorCritic

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--no-track", action="store_true", help="Disable MLflow logging")
    args = parser.parse_args()

    config = {
        "LR": 5e-5,
        "NUM_ENVS": 32,
        "NUM_STEPS": 2048,
        "TOTAL_TIMESTEPS": 5e7,
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 4,
        "GAMMA": 0.995,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.1,
        "ENT_COEF": 0.0001,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ACTIVATION": "tanh",
        "ANNEAL_LR": True,
        "RUN_NAME": "ppo_drone_final_9",
        "CKPT_DIR": "checkpoints",
        "DEBUG": True,
        "CHECKPOINT_FREQ": 5e6,
    }

    loaded_params = None
    if args.resume:
        if os.path.exists(args.resume):
            loaded_params = load_checkpoint(args.resume)
            print(f"Resuming training with weights from {args.resume}")
        else:
            raise FileNotFoundError(f"Checkpoint not found at {args.resume}")

    if not args.no_track:
        mlflow.set_experiment("ppo_drone")
        mlflow.start_run(run_name=config["RUN_NAME"])
        mlflow.log_params(config)
    else:
        print("MLflow tracking disabled.")

    try:
        rng = jax.random.PRNGKey(42)
        train_jit = jax.jit(make_train(config, initial_params=loaded_params))

        print("Starting training...")
        out = train_jit(rng)
        print("Training finished.")

        final_state = out["runner_state"][0]
        ckpt_path = save_checkpoint(final_state, config)

        if not args.no_track and mlflow.active_run():
            mlflow.log_artifact(ckpt_path)
            print(f"Logged checkpoint artifact: {ckpt_path}")

        eval_and_log_artifacts(final_state, config)

    finally:
        if mlflow.active_run():
            mlflow.end_run()

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray

def make_train(config, initial_params=None):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    env = DroneRaceEnv()
    env_params = DEFAULT_PARAMS

    env = FlattenObservationWrapper(env)
    env = LogWrapper(env)
    env = AutoResetWrapper(env)
    env = VecEnv(env)

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

        # RESUME FROM CHECKPOINT
        if initial_params is not None:
            train_state = train_state.replace(params=initial_params)

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

                obsv, env_state, reward, done, info = env.step(rng_step, env_state, action, env_params)

                transition = Transition(done, action, value, reward, log_prob, last_obs, info)
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

            # DEBUG & MLFLOW LOGGING
            def logging_callback(info, loss_info, update_iter):
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
                    f"Ret: {avg_ret:<7.1f} | "
                    f"Len: {avg_len:<5.0f} | "
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

            jax.debug.callback(logging_callback, metric, loss_info, update_i)

            # CHECKPOINTING
            if config.get("CHECKPOINT_FREQ") is not None:
                steps_per_update = config["NUM_STEPS"] * config["NUM_ENVS"]
                ckpt_interval_updates = max(1, int(config["CHECKPOINT_FREQ"] // steps_per_update))

                def save_callback(state, update_count):
                    step_num = int(update_count) * steps_per_update
                    temp_config = config.copy()
                    temp_config["RUN_NAME"] = f"{config['RUN_NAME']}_step{step_num}"
                    print(f"Saving checkpoint at step {step_num}...")
                    save_checkpoint(state, temp_config)

                is_save_step = ((update_i + 1) % ckpt_interval_updates == 0)

                jax.lax.cond(
                    is_save_step,
                    lambda args: jax.debug.callback(save_callback, *args),
                    lambda _: None,
                    (train_state, update_i + 1)
                )

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
    main()

