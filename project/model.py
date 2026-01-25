"""
Task 1: Learned Dynamics Model

You will implement a JAX dynamics model that approximates the
black-box transition in "acro_step.bin".

Required signature:
    x_next = step(x, u, dt, params)

Conventions (course):
- World frame: x forward, y left, z up (right-handed).
- Quaternion order: [qw, qx, qy, qz].
- State x shape (21,):
    [0:3]   position (m)
    [3:6]   velocity (m/s)
    [6:9]   acceleration (m/s^2)
    [9:13]  orientation quaternion (unit)
    [13:16] angular velocity (rad/s), body frame
    [16:20] previous applied action
    [20]    battery voltage (V), valid range [22, 24]
- Action u shape (4,) in [-1, 1]:
    [u_roll, u_pitch, u_yaw, u_thrust]

Minimum model structure (must implement):
1) First-order lag on commands
2) Linear mapping from RC rates to body rates using fixed max_rate
3) Polynomial thrust curve T(u_thrust, V) (fit coefficients)
4) Translational dynamics with thrust + gravity (drag optional)

This file is incomplete, fill in the TODOs.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

# Quaternion helpers provided in the handout (utils.py)
from utils import quat_mul, quat_normalize, quat_rotate


class ModelParameters(NamedTuple):
    # Learnable
    tau: jax.Array  # shape (4,), > 0  (roll, pitch, yaw, thrust) delay constants [s]
    thrust_coeffs: jax.Array  # shape (6,) thrust poly coeffs for [1, u, u^2, u^3, V, u*V]

    # Fixed
    max_rate: jax.Array  # shape (3,) deg/s = (618.0, 618.0, 120.0)
    m: float  # kg
    g: float  # m/s^2


DEFAULT_PARAMS = ModelParameters(
    tau=jnp.array([0.017000000923871994, 0.017000000923871994, 0.029929455369710922, 0.025], dtype=jnp.float32),  # TODO: replace with fitted values
    # tau=jnp.array([0.1, 0.5, 0.9, 0.01], dtype=jnp.float32),  # try
    # tau=jnp.array([0.03020575, 0.03020575, 0.01274104, 0.01], dtype=jnp.float32),  # try
    # tau=jnp.array([0.01, 0.01, 0.01, 0.01], dtype=jnp.float32),  # original values
    thrust_coeffs=jnp.array([
        1.,
        1., 
        1., 
        1.,
        1.,
        1.,
        ], dtype=jnp.float32),  # guess
    # thrust_coeffs=jnp.ones((6,), dtype=jnp.float32),  # TODO: replace with fitted values

    max_rate=jnp.array([618.0, 618.0, 120.0], dtype=jnp.float32),
    m=1.0,
    g=9.80665,
)


def thrust_polynomial(u_thrust: jax.Array, battery_v: jax.Array, coeffs: jax.Array) -> jax.Array:
    """
    Voltage-dependent thrust polynomial:
    """
    # c0 + c1*uth + c2*uth^2 + c3*uth^3 + c4*V + c5*(uth*V)
    # Hints:
    # - Keep types as float32.
    u = jnp.array([1.0, u_thrust, u_thrust ** 2, u_thrust ** 3, battery_v, u_thrust * battery_v], dtype=jnp.float32)
    return jnp.dot(coeffs, u)


def first_order_delay(u_prev_applied: jax.Array, u_cmd: jax.Array, dt: float, tau: jax.Array) -> jax.Array:
    """
    First-order lag on controls.
    """
    # Hints:
    # - tau is shape (4,) and strictly positive.
    # - alpha should be shape (4,) and computed elementwise.
    alpha = 1.0 - jnp.exp(-dt / tau)
    u_delayed = u_prev_applied + alpha * (u_cmd - u_prev_applied)
    return u_delayed


def map_rates(u_delayed: jax.Array, max_rate: jax.Array) -> jax.Array:
    """
    Map delayed commands to body rates.
    """
    w_cmd = u_delayed[0:3] * max_rate * (jnp.pi / 180.0)  # rad/s
    return w_cmd

@jax.jit
def step(x: jax.Array, u: jax.Array, dt: float, params: ModelParameters) -> jax.Array:
    """Learned transition: x_{t+1} = f_hat(x_t, u_t, dt; params)

    TODO (students): implement the minimum viable model:
      1) Decode x into pos, vel, q, prev_action, battery.
      2) Apply first-order delay: u_delayed = first_order_delay(prev_action, u, dt, params.tau)
      3) Map rates:
            body_rates = ...
      4) Quaternion integration (Euler step):
            q_next = ...
      5) Thrust:
            T = thrust_polynomial(u_delayed[3], battery, params.thrust_coeffs)
            a_thrust_world = ...
      6) Integrate state based on computed acceleration
      7) Update state fields:
            - acceleration = a
            - orientation = q_next
            - angular velocity = body_rates
            - previous action = u_delayed
            - battery: keep constant (optionally model a slow discharge)

    Return x_next with the same layout as x.
    """
    x = jnp.asarray(x, dtype=jnp.float32)
    u = jnp.asarray(u, dtype=jnp.float32)

    # 1) Decode state
    pos = x[0:3]
    vel = x[3:6]
    q = x[9:13]
    u_prev = x[16:20]
    battery = x[20]

    # 2) First-order delay
    u_delayed = first_order_delay(u_prev, u, dt, params.tau)

    # 3) Map rates
    body_rates = map_rates(u_delayed, params.max_rate)

    # 4) Quaternion integration (Euler step)
    q_next = quat_mul(q, jnp.array([0.0, body_rates[0], body_rates[1], body_rates[2]]) * dt / 2.0)
    q_next = quat_normalize(q + q_next)

    # 5) Thrust
    T = thrust_polynomial(u_delayed[3], battery, params.thrust_coeffs)

    # 6) Integrate state based on computed acceleration
    # TODO: check if we need q or q_next here
    a_thrust_world = quat_rotate(q, jnp.array([0.0, 0.0, (T / params.m) - params.g]))
    vel_next = vel + a_thrust_world * dt
    pos_next = pos + vel_next * dt

    # X) battery discharge 1V per minute
    battery = battery - (dt / 60.0)  # optional
    # 7) Update state fields
    x_next = jnp.concatenate([
        pos_next,
        vel_next,
        a_thrust_world,
        q_next,
        body_rates,
        u_delayed,
        battery, # TODO optionally model battery discharge
    ], axis=None) # flatten and then concatenate


    return x_next


if __name__ == "__main__":
    x0 = jnp.array(
        [
            0.0, 0.0, 0.0,  # pos
            0.0, 0.0, 0.0,  # vel
            0.0, 0.0, 0.0,  # acc
            1.0, 0.0, 0.0, 0.0,  # quat
            0.0, 0.0, 0.0,  # body rates
            0.0, 0.0, 0.0, -1.0,  # prev action
            24.0,  # battery
        ],
        dtype=jnp.float32,
    )
    u0 = jnp.array([0.0, 0.0, 0.0, -1.0], dtype=jnp.float32)
    x1 = step(x0, u0, 0.01, DEFAULT_PARAMS)
    print(x1)
