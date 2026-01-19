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
    tau=jnp.array([0.01, 0.01, 0.01, 0.01], dtype=jnp.float32),  # TODO: replace with fitted values
    thrust_coeffs=jnp.zeros((6,), dtype=jnp.float32),  # TODO: replace with fitted values

    max_rate=jnp.array([618.0, 618.0, 120.0], dtype=jnp.float32),
    m=1.0,
    g=9.80665,
)


def thrust_polynomial(u_thrust: jax.Array, battery_v: jax.Array, coeffs: jax.Array) -> jax.Array:
    """
    Voltage-dependent thrust polynomial:
    """
    # TODO: implement as hinted in exercise sheet.
    # Hints:
    # - Keep types as float32.
    raise NotImplementedError("TODO: implement thrust_polynomial")


def first_order_delay(u_prev_applied: jax.Array, u_cmd: jax.Array, dt: float, tau: jax.Array) -> jax.Array:
    """
    First-order lag on controls.
    """
    # TODO: implement first-order lag.
    # Hints:
    # - tau is shape (4,) and strictly positive.
    # - alpha should be shape (4,) and computed elementwise.
    raise NotImplementedError("TODO: implement first_order_delay")


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

    # Decode state
    pos = x[0:3]
    vel = x[3:6]
    q = x[9:13]
    u_prev = x[16:20]
    battery = x[20]

    raise NotImplementedError("TODO: implement step()")


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
    x1 = step(x0, u0, 0.01, DEFAULT_PARAMS)  # will raise until implemented
    print(x1)
