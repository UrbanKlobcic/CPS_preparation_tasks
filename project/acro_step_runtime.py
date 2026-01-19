# -*- coding: utf-8 -*-
"""
Runtime loader for the exported Acro transition artifact.

This module loads the serialized JAX export blob and exposes:

    step(x, u) -> x_next

Where:
- x is the 18D state vector
- u is the 4D action vector
- x_next is the 18D next state

The dt and all model parameters are hardcoded inside the binary export.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

jax.config.update("jax_platforms", "cpu")  # Important! Binary is CPU-only

# Default artifact location
DEFAULT_BIN = Path(__file__).with_name("acro_step.bin")

# Global ref to loaded function
LOADED_FUNC = None

# Valid Start State
DEFAULT_STATE = jnp.array([
    0.0, 0.0, 0.0,  # Position (m)
    0.0, 0.0, 0.0,  # Velocity (m/s)
    0.0, 0.0, 0.0,  # Acceleration (m/s^2)
    1.0, 0.0, 0.0, 0.0,  # Quaternion orientation
    0.0, 0.0, 0.0,  # Angular velocity (rad/s)
    0.0, 0.0, 0.0, -1.0,  # Previous action
    24.0  # Battery voltage (V)
], dtype=jnp.float32)


@functools.lru_cache(maxsize=1)
def _load_exported(bin_path: str) -> Any:
    blob = Path(bin_path).read_bytes()
    return jax.export.deserialize(blob)


def step(x, u, *, bin_path: str | Path = DEFAULT_BIN):
    """
    Apply the exported transition function to get the next state.

    Args:
      x: state array-like, shape (21,), float32
      u: action array-like, shape (4,), float32
      bin_path: path to the exported binary artifact

    State x:
      [x, y, z,                                                 Position (m)
       vx, vy, vz,                                              Velocity (m/s)
       ax, ay, az,                                              Acceleration (m/s^2)
       qw, qx, qy, qz,                                          Quaternion orientation
       wx, wy, wz,                                              Angular velocity (rad/s)
       u_roll_prev, u_pitch_prev, u_yaw_prev, u_thrust_prev,    Previous action
       battery_V]                                               Battery voltage (V)
    Action u:
      [u_roll, u_pitch, u_yaw, u_thrust]  all in [-1, 1]

    Returns:
      Next state, shape (21,)
    """
    global LOADED_FUNC

    if LOADED_FUNC is None:
        LOADED_FUNC = _load_exported(str(bin_path))

    x_j = jnp.asarray(x, dtype=jnp.float32)
    u_j = jnp.asarray(u, dtype=jnp.float32)

    # exported.call triggers compilation on first use, then runs the compiled executable
    y = LOADED_FUNC.call(x_j, u_j)

    return y


def print_state(x):
    """
    Pretty-print the state vector.

    Args:
      x: state array-like, shape (21,), float32
    """
    x = jnp.asarray(x, dtype=jnp.float32)
    print(f"Position:       {x[0:3]}")
    print(f"Velocity:       {x[3:6]}")
    print(f"Acceleration:   {x[6:9]}")
    print(f"Orientation:    {x[9:13]}")
    print(f"Angular Vel.:   {x[13:16]}")
    print(f"Prev. Action:   {x[16:20]}")
    print(f"Battery Volt.:  {x[20]}")


if __name__ == '__main__':
    initial_state = DEFAULT_STATE
    action = jnp.array([0.0, 0.0, 0.0, -1.0], dtype=jnp.float32)

    next_state = step(initial_state, action)

    print("Initial state:")
    print_state(initial_state)
    print("Action:", action)
    print("Next state:")
    print_state(next_state)
