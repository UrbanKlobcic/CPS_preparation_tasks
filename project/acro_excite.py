
from __future__ import annotations
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from pathlib import Path
from typing import NamedTuple
import acro_step_runtime
from model import step as model_step, ModelParameters, DEFAULT_PARAMS


AXIS_NAMES = ['roll', 'pitch', 'yaw', 'thrust']


class ExcitationLog(NamedTuple):
    u: jnp.Array  # shape (T, 4)
    b: jnp.Array  # shape (T, 1)
    p: jnp.Array  # shape (T, 3)
    w: jnp.Array  # shape (T, 3)
    v: jnp.Array  # shape (T, 1)
    a: jnp.Array  # shape (T, 3)
    q: jnp.Array  # shape (T, 4)


def excite_model(
    u: jnp.Array,
    model: bool = True,
    dt: float = 0.01,
    params: ModelParameters = DEFAULT_PARAMS,
    initial_state: jax.Array = acro_step_runtime.DEFAULT_STATE,
    reset_velocity: bool = True,
) -> jnp.Array:
    """
    Excite the model with given commands, log predicted body rates, world frame acceleration, and battery voltage.

    Args:
        u: array of shape (T, 4) with commands
        model: if True, use the learned model; if False, use the blackbox

    Returns:
        [w, a, b] as one array with shape (T, 7)
    """
    steps = u.shape[0]
    x = initial_state

    def blackbox_step(x, u, _, __) -> jnp.Array:
        return acro_step_runtime.step(x, u)

    if model:
        step_fn = model_step
    else:
        step_fn = blackbox_step

    def reset_fn(x):
        return x.at[3:6].set(jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32))

    def no_reset_fn(x):
        return x

    def step(carry, t):
        x = carry
        # reset velocity to zero for both models to avoid influence of drag if we get too fast
        # if reset_velocity:
        #     x = x.at[3:6].set(jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32))
        condition = reset_velocity & ((x[3] > 20.0) | (x[4] > 20.0) | (x[5] > 20.0))
        x = jax.lax.cond(condition, reset_fn, no_reset_fn, x)

        u_t = u[t]

        # Predict with model
        x_next = step_fn(x, u_t, dt, params)
        pos_pred = x_next[0:3]  # Position (m)
        vel_pred = x_next[3:6]  # Velocity (m/s)
        acc_pred = x_next[6:9]  # world frame acceleration (m/s^2)
        q_pred = x_next[9:13]  # Quaternion orientation
        w_pred = x_next[13:16]  # body rates / Angular velocity (rad/s)
        u_prev = x_next[16:20]  # Previous action
        bat_pred = x_next[20]  # battery voltage

        return x_next, x_next  # jnp.concatenate([w_pred, acc_pred, bat_pred], axis=None)

    # collect model responses
    x_final, result = jax.lax.scan(
        step,
        x,
        jnp.arange(steps)
    )
    # print("Final state after excitation:", x_final)
    return result


def generate_dirac_function(time, freq, amplitude):
    return amplitude * (jnp.abs(time - 0.5) < (dt / 2)).astype(jnp.float32)


def generate_constant_function(time, freq, amplitude):
    return amplitude * jnp.ones_like(time)


def generate_sine_function(time, freq, amplitude):
    return amplitude * jnp.sin(2 * jnp.pi * freq * time)


def generate_learning_function(time, freq, amplitude):
    return amplitude * (
        jnp.heaviside(time - 0.019, 1.0) * jnp.heaviside(0.021 - time, 1.0)
        + (jnp.heaviside(time - 0.5, 1.0) * jnp.heaviside(0.75 - time, 1.0) - 1)
        + jnp.heaviside(time - 0.85, 1.0) * jnp.heaviside(1.0 - time, 1.0)
        + (jnp.heaviside(time - 1.0, 1.0) * jnp.sin(2 * jnp.pi * freq * time))
    )


def generate_test_function(time: jnp.Array, freq: float, amplitude: float) -> jnp.Array:
    # Chirp for frequency sweep
    chirp = jnp.sin(2 * jnp.pi * (freq * time + 0.5 * freq * time**2))

    # Square wave for sharp changes
    square = jnp.sign(jnp.sin(2 * jnp.pi * freq * time))

    signal = (
        (jnp.heaviside(time - 0.1, 2.0) * jnp.heaviside(0.2 - time, 2.0)) +
        -(jnp.heaviside(time - 0, 1.0) * jnp.heaviside(1.0 - time, 1.0)) +
        jnp.heaviside(time - 1.0, 1.0) * jnp.heaviside(2.0 - time, 1.0) * chirp +
        jnp.heaviside(time - 2.0, 1.0) * jnp.heaviside(4.0 - time, 1.0) * square +
        jnp.heaviside(time - 4.0, 1.0) * chirp * 1.5
    )

    return amplitude * signal


def generate_commands(
    dt: float,
    duration: float,
    axis: int,
    freq: float = 1.0,
    amplitude: float = 1.0,
    func: callable = generate_learning_function,
) -> jnp.Array:
    """
    Generate a command sequence for a given time and axis.

    Args:
        t: time (seconds)
        axis: 0=roll, 1=pitch, 2=yaw, 3=thrust
        freq: sine wave frequency (Hz)
        amplitude: sine amplitude in [-1, 1]
        func: function to generate command value given time, freq, amplitude
    """
    steps = int(duration / dt)

    def command_fn(_, t):
        time = t * dt
        u = jnp.array([0.0, 0.0, 0.0, -1.0], dtype=jnp.float32)
        u = u.at[axis].set(func(time, freq, amplitude))
        return _, u

    _, commands = jax.lax.scan(
        command_fn,
        None,
        jnp.arange(steps)
    )
    return commands


def thrust_agent(az: jnp.Array, m: float, g: float) -> jnp.Array:
    T = m * (az + g)
    return T


def fit_tau(log: ExcitationLog, axis: int, lr: float = 0.01, steps: int = 100,
            dt: float = 0.01, params: ModelParameters = DEFAULT_PARAMS) -> float:
    """
    Linear regression to fit tau for one axis by minimizing MSE between measured and predicted body rates.
    (Reference tutorial: https://coax.readthedocs.io/en/latest/examples/linear_regression/jax.html)

    Args:
        log: ExcitationLog with commands u and measured body rates w
        axis: 0=roll, 1=pitch, 2=yaw
        lr: learning rate
        steps: number of gradient descent steps
        dt: time step in seconds
        params: model parameters

    Returns:
        Fitted tau value for the given axis as float
    """

    def loss_fn(tau_t, log: ExcitationLog) -> jnp.Array:
        params_t = ModelParameters(
            tau=params.tau.at[axis].set(tau_t),
            thrust_coeffs=params.thrust_coeffs,
            max_rate=params.max_rate,
            m=params.m,
            g=params.g,
        )

        # do this as subfunction to capture the updated params without adding more arguments (because of scan)
        def forward(x, u_t):
            x_pred = model_step(x, u_t, dt, params_t)
            return x_pred, x_pred[13:16]

        _, w_pred_all = jax.lax.scan(
            forward,
            acro_step_runtime.DEFAULT_STATE,
            log.u
        )
        mse = jnp.mean(jnp.square(log.w - w_pred_all))
        return mse

    grad_fn = jax.grad(loss_fn)

    def update_fn(tau_val, _):
        grad = grad_fn(tau_val, log)
        tau_val = tau_val - lr * grad
        tau_val = jnp.clip(tau_val, 1e-6, 1.0)  # keep 0 < eps <= tau <= 1.0
        return tau_val, None

    tau_t = DEFAULT_PARAMS.tau[axis]
    tau_final, _ = jax.lax.scan(update_fn, tau_t, jnp.arange(steps))

    return float(tau_final)


def fit_thrust(log: ExcitationLog, params: ModelParameters, lr: float = 0.001, steps: int = 500, dt: float = 0.01) -> jnp.Array:
    """
    Fit thrust coefficients using nonlinear least squares.

    Args:
        log: ExcitationLog with commands u, measured acceleration a, and battery voltage b

    Returns:
        Fitted thrust coefficients as jnp.Array of shape (6,)
    """

    def residuals_fn(coeffs: jnp.Array, log: ExcitationLog) -> jnp.Array:
        def forward(x, u_t):
            x_pred = model_step(x, u_t, dt, params._replace(thrust_coeffs=coeffs))
            return x_pred, x_pred

        _, pred = jax.lax.scan(
            forward,
            acro_step_runtime.DEFAULT_STATE,
            log.u
        )

        # Estimate thrust from observations as given in the assignment
        T_pred = thrust_agent(pred[:, 8], params.m, params.g)
        T_meas = thrust_agent(log.a[:, 2], params.m, params.g)
        # jax.debug.print("T_meas: {T_meas_}, T_pred: {T_pred_}", T_meas_=T_meas[0:5], T_pred_=T_pred[0:5])
        # jax.debug.print("shape T_meas: {shape_meas_}, shape T_pred: {shape_pred_}", shape_meas_=T_meas.shape, shape_pred_=T_pred.shape)
        # jax.debug.print("shape log.a[:, 2]: {shape_meas_}, shape pred[:, 8]: {shape_pred_}", shape_meas_=log.a[:, 2].shape, shape_pred_=pred[:, 8].shape)
        # jax.debug.print("coeffs: {coeffs_}", coeffs_=coeffs)
        return T_meas - T_pred

    def loss_fn(coeffs: jnp.Array, log: ExcitationLog) -> jnp.Array:
        res = residuals_fn(coeffs, log)
        return jnp.mean(res ** 2)

    jacobian_fn = jax.jacfwd(residuals_fn)

    k = 1
    coeffs_init = jnp.array([k, k, k, k, k, k], dtype=jnp.float32)

    # Simple gradient descent
    coeffs = coeffs_init

    def update_fn(coeffs, _):
        # # usual version
        # grad = grad_fn(coeffs, log)
        # coeffs = coeffs - lr * grad

        # nonlinear least squares version
        r = residuals_fn(coeffs, log)  # shape (T,)
        J = jacobian_fn(coeffs, log)  # shape (T, 6)
        # Levenberg-Marquardt damping
        lambda_damping = 1e-3
        JT = J.transpose()
        JTJ = JT @ J + lambda_damping * jnp.eye(JT.shape[0])  # shape (6, 6)
        JTr = JT @ r  # shape (6,)
        update = jnp.linalg.solve(JTJ, JTr)
        coeffs = coeffs - lr * update
        return coeffs, None

    coeffs, _ = jax.lax.scan(update_fn, coeffs, jnp.arange(steps))

    return coeffs


def display_body_rates(log: ExcitationLog, axis: int, prediction: ExcitationLog | None = None, additional: jnp.Array = None, dt: float = 0.01):
    time = jnp.arange(log.u.shape[0]) * dt
    plt.subplot(3, 1, axis + 1)
    plt.plot(time, log.u[:, axis], label='Commands', color='green', linestyle='--')
    plt.plot(time, log.w[:, axis], label='Measured (blackbox)', color='blue')
    # plt.plot(time, log.w_predicted[:, axis], label='Predicted (initial guess)', color='orange', linestyle='--')
    if prediction is not None:
        plt.plot(time, prediction[:, axis], label='Predicted (fitted tau)', color='red', linestyle=':')
    if additional is not None:
        plt.plot(time, additional[:, axis], label='Additional', color='purple', linestyle='-.')
    plt.title(f'Body Rate {AXIS_NAMES[axis]} - Measured vs Predicted')
    plt.xlabel('Time (s)')
    plt.ylabel('Body Rate (rad/s)')
    plt.legend()
    plt.grid()


def display_thrust(log: ExcitationLog, volt: int, prediction: ExcitationLog | None = None, additional: jnp.Array = None, params: ModelParameters = DEFAULT_PARAMS, dt: float = 0.01):
    time = jnp.arange(log.u.shape[0]) * dt
    idx = 24-volt
    plt.subplot(3, 1, idx + 1)
    # plt.plot(time, log.a[:, 0], label='Measured x (blackbox)', color='red')
    # plt.plot(time, log.a[:, 1], label='Measured y (blackbox)', color='orange')
    plt.plot(time, thrust_agent(log.a[:, 2], params.m, params.g), label='Measured thrust (blackbox)', color='blue')
    if log.b is not None:
        plt.plot(time, log.b[:], label='Battery level (blackbox)', color='grey', linestyle=':')
    if prediction is not None and prediction.b is not None:
        plt.plot(time, prediction.b[:], label='Battery level (model)', color='lightgrey', linestyle='-.')
    # plt.plot(time, log.w_predicted[:, axis], label='Predicted (initial guess)', color='orange', linestyle='--')
    if prediction is not None:
        plt.plot(time, thrust_agent(prediction.a[:, 2], params.m, params.g), label='Predicted thrust (model)', color='red', linestyle=':')
    if additional is not None:
        plt.plot(time, additional[:, 2], label='Additional', color='purple', linestyle='-.')
    plt.plot(time, log.u[:, -1], label='Commands', color='green', linestyle='--')
    plt.title(f'Thrust at Voltage {volt} - Measured vs Predicted')
    plt.xlabel('Time (s)')
    plt.ylabel('Thrust (N)')
    plt.legend()
    plt.grid()


def print_mse(axis: int, blackbox_obs_test: jax.Array, model_obs_test: jax.Array, params: ModelParameters):
    mse_p = jnp.mean(jnp.square(blackbox_obs_test[:, 0:3] - model_obs_test[:, 0:3]))
    print(f"Axis {AXIS_NAMES[axis]} MSE position: {mse_p:.6f}")
    mse_v = jnp.mean(jnp.square(blackbox_obs_test[:, 3:6] - model_obs_test[:, 3:6]))
    print(f"Axis {AXIS_NAMES[axis]} MSE velocity: {mse_v:.6f}")
    mse_w = jnp.mean(jnp.square(blackbox_obs_test[:, 13:16] - model_obs_test[:, 13:16]))
    print(f"Axis {AXIS_NAMES[axis]} MSE body rates: {mse_w:.6f}")
    mse_q = jnp.mean(jnp.square(blackbox_obs_test[:, 9:13] - model_obs_test[:, 9:13]))
    print(f"Axis {AXIS_NAMES[axis]} MSE quaternion: {mse_q:.6f}")
    mse_a = jnp.mean(jnp.square(
        thrust_agent(blackbox_obs_test[:, 6:9], params.m, params.g)
        - thrust_agent(model_obs_test[:, 6:9], params.m, params.g)))
    print(f"Axis {AXIS_NAMES[axis]} MSE thrust: {mse_a:.6f}")


def plot_aux_3d(*args, dt: float = 0.01, title: str = "Commands"):
    """
    Plot additional data over time.

    Args:
        data: array of shape (T, N) with auxiliary data
        title: title for the plot
    """
    time = jnp.arange(args[0][0].shape[0]) * dt

    ax = plt.subplot(224)
    print(f"auxiliary data shape: {len(args)}{args[0][0].shape}")
    for data in args:
        ax.plot(time, data[0], label=data[1])

    # Labels and title
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Command Value')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)


def plot_commands_3d(commands: jnp.Array, dt: float = 0.01, title: str = "Commands"):
    """
    Plot the 3D commands over time.

    Args:
        commands: array of shape (T, 4) with commands [roll, pitch, yaw, thrust]
        title: title for the plot
    """
    time = jnp.arange(commands.shape[0]) * dt

    print(f"commands shape: {commands.shape}")
    ax = plt.subplot(223)
    ax.plot(time, commands[:, 0], 'm-', label='Roll Command')
    ax.plot(time, commands[:, 1], 'g-', label='Pitch Command')
    ax.plot(time, commands[:, 2], 'b-', label='Yaw Command')
    ax.plot(time, commands[:, 3], 'r-', label='Thrust Command')

    # Labels and title
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Command Value')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)


def plot_trajectory_3d(commands: jnp.Array, model: bool = False,
                       params: ModelParameters = DEFAULT_PARAMS,
                       initial_state: jax.Array = acro_step_runtime.DEFAULT_STATE,
                       dt: float = 0.01, title: str = "Drone Trajectory"):
    """
    Execute a command sequence and display the resulting 3D position trajectory.

    Args:
        commands: array of shape (T, 4) with commands [roll, pitch, yaw, thrust]
        model: if True, use the learned model; if False, use the blackbox
        params: model parameters
        initial_state: initial state vector
        dt: time step in seconds
        title: title for the plot
    Returns:
        result: array of shape (T, 21) with state vectors over time
    """
    # Execute the command sequence
    result = excite_model(u=commands, model=model, dt=dt, params=params, initial_state=initial_state, reset_velocity=False)

    # Extract positions from result (first 3 elements of state vector)
    positions = result[:, 0:3]

    # Create 3D plot
    sub = 222 if model else 221
    ax = plt.subplot(sub, projection='3d')

    # Plot trajectory
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=2, label='Trajectory')

    # Mark start and end points
    ax.scatter([positions[0, 0]], [positions[0, 1]], [positions[0, 2]],
               color='green', s=100, marker='o', label='Start', zorder=5)
    ax.scatter([positions[-1, 0]], [positions[-1, 1]], [positions[-1, 2]],
               color='red', s=100, marker='s', label='End', zorder=5)

    # Labels and title
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_zlabel('Z Position (m)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)

    return result


def onestep_prediction_errors(
    commands: jnp.Array,
    params: ModelParameters = DEFAULT_PARAMS,
    initial_state: jax.Array = acro_step_runtime.DEFAULT_STATE,
    dt: float = 0.01
) -> dict:
    """
    Calculate one-step prediction errors by comparing model predictions to actual observations.
    
    For each timestep t, uses state x_t and command u_t to predict x_{t+1}, then compares
    to the observed x_{t+1}.

    Args:
        commands: array of shape (T, 4) with commands
        model: if True, use the learned model; if False, use the blackbox
        params: model parameters
        initial_state: initial state vector
        dt: time step in seconds

    Returns:
        Dictionary containing:
        - 'errors': shape (T-1, 21) - full state prediction errors
        - 'mse_position': float - MSE for position (first 3 states)
        - 'mse_velocity': float - MSE for velocity (states 3-6)
        - 'mse_acceleration': float - MSE for acceleration (states 6-9)
        - 'mse_quaternion': float - MSE for quaternion (states 9-13)
        - 'mse_body_rates': float - MSE for body rates (states 13-16)
        - 'mse_thrust': float - MSE for thrust (derived from acceleration)
    """
    steps = commands.shape[0]
    x = initial_state
    
    def blackbox_step(x, u, _, __) -> jnp.Array:
        return acro_step_runtime.step(x, u)

    def reset_fn(x):
        return x.at[3:6].set(jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32))

    def no_reset_fn(x):
        return x
    
    def step(carry, t):
        x_current = carry
        u_t = commands[t]
        
        x_next = blackbox_step(x_current, u_t, dt, params)
        x_pred = model_step(x_current, u_t, dt, params)
        
        return x_next, (x_pred, x_next)

    _, (x_pred, x_nexts) = jax.lax.scan(step, x, jnp.arange(steps))
    
    errors = x_nexts - x_pred
    
    # Compute MSE for different state components
    mse_position = float(jnp.mean(jnp.square(errors[:, 0:3])))
    mse_velocity = float(jnp.mean(jnp.square(errors[:, 3:6])))
    mse_acceleration = float(jnp.mean(jnp.square(errors[:, 6:9])))
    mse_quaternion = float(jnp.mean(jnp.square(errors[:, 9:13])))
    mse_body_rates = float(jnp.mean(jnp.square(errors[:, 13:16])))
    
    # Calculate thrust MSE from acceleration errors
    thrust_meas = thrust_agent(x_nexts[:, 8], params.m, params.g)
    thrust_pred = thrust_agent(x_pred[:, 8], params.m, params.g)
    mse_thrust = float(jnp.mean(jnp.square(thrust_meas - thrust_pred)))
    
    return {
        'errors': errors,
        'mse_position': mse_position,
        'mse_velocity': mse_velocity,
        'mse_acceleration': mse_acceleration,
        'mse_quaternion': mse_quaternion,
        'mse_body_rates': mse_body_rates,
        'mse_thrust': mse_thrust,
    }


def print_onestep_errors(error_dict: dict, label: str = ""):
    """Print one-step prediction error metrics."""
    print(f"\n{'='*50}")
    print(f"One-Step Prediction Errors {label}")
    print(f"{'='*50}")
    print(f"MSE Position:     {error_dict['mse_position']:.8f}")
    print(f"MSE Velocity:     {error_dict['mse_velocity']:.8f}")
    print(f"MSE Acceleration: {error_dict['mse_acceleration']:.8f}")
    print(f"MSE Quaternion:   {error_dict['mse_quaternion']:.8f}")
    print(f"MSE Body Rates:   {error_dict['mse_body_rates']:.8f}")
    print(f"MSE Thrust:       {error_dict['mse_thrust']:.8f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    print("Exciting model and fitting parameters...")
    params = DEFAULT_PARAMS
    fitted_taus = []
    learned_taus = []
    d = 5.0
    f = 1.0
    a = 1.0
    dt = 0.01

    do_tau = True
    do_thrust = True
    do_trajectory = True

    if do_tau:
        print("Fitting taus on learning signal...")
        fig_tau = plt.figure(1)
        for axis in range(3):  # roll, pitch, yaw (fix thrust to -1)
            print(f"\n--- Axis {AXIS_NAMES[axis]} ---")
            commands_test = generate_commands(dt=dt, duration=d, axis=axis, freq=f, amplitude=a)
            blackbox_obs = excite_model(u=commands_test, model=False)
            print(f"{blackbox_obs.shape=}")

            learned_tau = fit_tau(ExcitationLog(u=commands_test, w=blackbox_obs[:, 13:16], v=None, a=None, p=None, q=None, b=None),
                                  axis, lr=0.001, steps=50)
            learned_taus.append(learned_tau)
            print(f"Learned tau[{AXIS_NAMES[axis]}]: {learned_tau:.6f}")

        # set average tau for thrust axis, as this will be dominated by the coeffitients
        learned_taus.append(float(jnp.mean(jnp.array(learned_taus))))

        print("\nValidating learned taus on test signal...")
        params = ModelParameters(
            tau=jnp.array(learned_taus, dtype=jnp.float32),
            thrust_coeffs=DEFAULT_PARAMS.thrust_coeffs,
            max_rate=DEFAULT_PARAMS.max_rate,
            m=DEFAULT_PARAMS.m,
            g=DEFAULT_PARAMS.g,
        )
        for axis in range(3):

            commands_test = generate_commands(dt=dt, duration=d, axis=axis, freq=f*2, amplitude=a/2, func=generate_test_function)
            blackbox_obs_test = excite_model(u=commands_test, model=False)
            model_obs_test = excite_model(u=commands_test, model=True, params=params)

            # Log MSE after fitting
            print_mse(axis, blackbox_obs_test, model_obs_test, params)
            display_body_rates(
                ExcitationLog(u=commands_test, w=blackbox_obs_test[:, 13:16], v=None, a=None, p=None, q=None, b=None),
                axis,
                prediction=model_obs_test[:, 13:16],
            )

        fig_tau.show()
        print("\n")
        print("*" * 40)
        print(f"Learned tau values: {learned_taus}")
        print("*" * 40)

    if do_thrust:
        print("Fitting thrust coefficients...")
        fig_thrust = plt.figure(2)

        commands_learn = generate_commands(dt=dt, duration=60, axis=3, freq=f, amplitude=a)
        blackbox_obs = excite_model(u=commands_learn, model=False)
        learn_log = ExcitationLog(u=commands_learn, a=blackbox_obs[:, 6:9], v=None, w=blackbox_obs[:, 13:16], p=None, q=None, b=blackbox_obs[:, 20])
        fitted_coeffs = fit_thrust(learn_log, params, steps=100)
        print("*" * 40)
        print(f"Fitted thrust coefficients: {fitted_coeffs.tolist()}")
        print("*" * 40)
        params = params._replace(thrust_coeffs=fitted_coeffs)

        print("\nValidating fitted thrust coefficients on test signal...")
        for volt in range(24, 21, -1):
            print(f"\n--- Battery Voltage: {volt}V ---")
            commands_test = generate_commands(dt=dt, duration=4, axis=3, freq=f*2, amplitude=a, func=generate_test_function)
            x_initial = acro_step_runtime.DEFAULT_STATE.at[20].set(float(volt))
            blackbox_obs_test = excite_model(u=commands_test, model=False, initial_state=x_initial, reset_velocity=False)
            model_obs_test = excite_model(u=commands_test, model=True, params=params, initial_state=x_initial, reset_velocity=False)

            print_mse(3, blackbox_obs_test, model_obs_test, params)
            display_thrust(
                ExcitationLog(u=commands_test, a=blackbox_obs_test[:, 6:9], v=None, w=None, p=None, q=None, b=None),
                volt,
                prediction=ExcitationLog(u=commands_test, a=model_obs_test[:, 6:9], v=None, w=None, p=None, q=None, b=model_obs_test[:, 20]),
            )

        fig_thrust.show()

    if do_trajectory:
        print("\nPlotting trajectory...")
        x_initial = acro_step_runtime.DEFAULT_STATE.at[0:3].set(jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32))
        # plot demo trajectory
        fig_trajectory = plt.figure(3)

        commands = generate_commands(dt=0.01, duration=5, axis=3, freq=1.0, amplitude=1.0,
                                     func=generate_sine_function
                                     # func=generate_constant_function
                                     # func=generate_dirac_function
                                     )
        commands = commands.at[:, 0].set(generate_sine_function(jnp.arange(commands.shape[0]) * dt, freq=1.1, amplitude=1))
        commands = commands.at[:, 1].set(generate_sine_function(jnp.arange(commands.shape[0]) * dt, freq=1.2, amplitude=1))
        commands = commands.at[:, 2].set(generate_sine_function(jnp.arange(commands.shape[0]) * dt, freq=1.3, amplitude=1))
        x_black = plot_trajectory_3d(commands, model=False, title="Blackbox Trajectory", params=params, initial_state=x_initial)
        x_model = plot_trajectory_3d(commands, model=True, title="Model Trajectory", params=params, initial_state=x_initial)
        plot_commands_3d(commands, title="Commands")
        plot_aux_3d((x_black[:, 0], "Blackbox X"),
                    (x_black[:, 1], "Blackbox Y"),
                    (x_black[:, 2], "Blackbox Z"),
                    (x_model[:, 0], "Model X"),
                    (x_model[:, 1], "Model Y"),
                    (x_model[:, 2], "Model Z"),
                    title="Auxiliary Data"
                    )
        fig_trajectory.suptitle("Drone Trajectory Comparison")

    commands_test = generate_commands(dt=dt, duration=5, axis=0, freq=1.0, amplitude=1.0)
    errors_model = onestep_prediction_errors(
        commands_test, params=params
    )
    print_onestep_errors(errors_model, label="(Learned Model)")

    plt.show()
