"""
This file includes helper code for drawing racing gates and a simple drone body model.
Only the gate drawing portion of `visualize_state_action_sequence` is implemented.

Complete the remaining TODOs to visualize:
- the drone trajectory
- the drone pose over time (position + orientation)
- action and kinematic time-series plots

Expected input:
  sequence: list of (x, u) pairs
    x: shape (21,) float32
    u: shape (4,)  float32

State layout (x):
  [x, y, z,                                                 Position (m)
   vx, vy, vz,                                              Velocity (m/s)
   ax, ay, az,                                              Acceleration (m/s^2)
   qw, qx, qy, qz,                                          Quaternion orientation
   wx, wy, wz,                                              Angular velocity (rad/s)
   u_roll_prev, u_pitch_prev, u_yaw_prev, u_thrust_prev,    Previous action
   battery_V]                                               Battery voltage (V)

Action layout (u):
  [u_roll, u_pitch, u_yaw, u_thrust]  all in [-1, 1]

Gate input:
  gates: np.ndarray, shape (N, >=8), each row:
    [gate_id, cx, cy, cz, qw, qx, qy, qz, ...]
"""

from __future__ import annotations

import numpy as np
import rerun as rr


# Quaternion and rotation helpers (NumPy)
def _quat_rotate_np(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Rotate vector v by unit quaternion q = [w, x, y, z].
    """
    q = np.asarray(q, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)

    w, x, y, z = q
    q_vec = np.array([x, y, z], dtype=np.float32)

    t = 2.0 * np.cross(q_vec, v)
    return v + w * t + np.cross(q_vec, t)


def _euler_to_rotmat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Convert roll, pitch, yaw (rad) to a 3x3 rotation matrix.

    Convention: R = Rz(yaw) * Ry(pitch) * Rx(roll)
    (standard aerospace roll–pitch–yaw, body -> world).
    """
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    Rz = np.array(
        [
            [cy, -sy, 0.0],
            [sy, cy, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    Ry = np.array(
        [
            [cp, 0.0, sp],
            [0.0, 1.0, 0.0],
            [-sp, 0.0, cp],
        ],
        dtype=np.float32,
    )
    Rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cr, -sr],
            [0.0, sr, cr],
        ],
        dtype=np.float32,
    )

    # body -> world
    return (Rz @ Ry @ Rx).astype(np.float32)


# Gate geometry helpers
def _make_gate_loops(
        center: np.ndarray,
        quat: np.ndarray,
        outer_size: float = 2.7,
        inner_size: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build outer and inner rectangular loops (in world frame) for a gate.

    - `center`: [3] gate center in world frame
    - `quat`:   [4] quaternion [w, x, y, z] for gate orientation
    - `outer_size`: outer square side (m)
    - `inner_size`: inner square opening side (m)
    """

    def square_points(size: float) -> np.ndarray:
        h = size * 0.5
        # In gate-local frame: plane at x=0, spanning y/z.
        return np.array(
            [
                [0.0, -h, -h],
                [0.0, h, -h],
                [0.0, h, h],
                [0.0, -h, h],
            ],
            dtype=np.float32,
        )

    outer_local = square_points(outer_size)
    inner_local = square_points(inner_size)

    def to_world(pts_local: np.ndarray) -> np.ndarray:
        return np.stack([_quat_rotate_np(quat, p) + center for p in pts_local], axis=0)

    outer_world = to_world(outer_local)
    inner_world = to_world(inner_local)

    # Close the loops by repeating the first point at the end
    outer_loop = np.concatenate([outer_world, outer_world[:1]], axis=0)
    inner_loop = np.concatenate([inner_world, inner_world[:1]], axis=0)

    return outer_loop, inner_loop


# Drone body model (in body frame)
_FRONT_SPAN = 0.18  # front motors Y distance (m)
_BACK_SPAN = 0.28  # rear motors Y distance (m)
_FB_DIST = 0.30  # distance between front/back motor lines (m)

_front_x = +_FB_DIST / 2.0
_back_x = -_FB_DIST / 2.0
_front_y_off = _FRONT_SPAN / 2.0
_back_y_off = _BACK_SPAN / 2.0

# Indexing: 0=front_left, 1=front_right, 2=back_left, 3=back_right
_DRONE_MOTORS_BODY = np.array(
    [
        [_front_x, +_front_y_off, 0.0],  # front-left
        [_front_x, -_front_y_off, 0.0],  # front-right
        [_back_x, +_back_y_off, 0.0],  # rear-left
        [_back_x, -_back_y_off, 0.0],  # rear-right
    ],
    dtype=np.float32,
)


def _drone_body_points_world(
        pos_world: np.ndarray,
        euler_rad: np.ndarray,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    """
    Given drone position [3] in world and Euler angles [roll, pitch, yaw] (rad),
    return:

      - motors_world: (4, 3) motor positions in world frame
      - arms: list of 2x3 arrays for X-shaped arms (two line segments)
      - cam_dir_world: (3,) camera direction in world frame
    """
    roll, pitch, yaw = float(euler_rad[0]), float(euler_rad[1]), float(euler_rad[2])
    R = _euler_to_rotmat(roll, pitch, yaw)  # body -> world

    motors_world = (R @ _DRONE_MOTORS_BODY.T).T + pos_world[None, :]

    # Arms as X:
    arm1 = np.stack([motors_world[0], motors_world[3]], axis=0)
    arm2 = np.stack([motors_world[1], motors_world[2]], axis=0)

    # Camera direction in body frame: forward + pitched up 40°
    cam_pitch_rad = np.deg2rad(40.0)
    cam_dir_body = np.array(
        [np.cos(cam_pitch_rad), 0.0, np.sin(cam_pitch_rad)],
        dtype=np.float32,
    )
    cam_dir_world = (R @ cam_dir_body).astype(np.float32)

    return (
        motors_world.astype(np.float32),
        [arm1.astype(np.float32), arm2.astype(np.float32)],
        cam_dir_world,
    )


# quaternion -> Euler conversion
def quat_to_euler_rpy(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion [w, x, y, z] to roll/pitch/yaw (rad).

    Convention: R = Rz(yaw) * Ry(pitch) * Rx(roll)  (body -> world)
    """
    q = np.asarray(q, dtype=np.float32).reshape(4, )
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])

    # Normalize
    n = np.sqrt(w * w + x * x + y * y + z * z)
    if n > 0.0:
        w, x, y, z = w / n, x / n, y / n, z / n

    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0, 1.0)  # avoid NaNs
    pitch = np.arcsin(sinp)

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw], dtype=np.float32)


# Small scalar logging helper
def _log_scalar(path: str, value: float) -> None:
    """Compat shim: log a single scalar using rr.Scalars."""
    rr.log(path, rr.Scalars(np.array([value], dtype=np.float32)))


# Visualization function to be implemented
def visualize_state_action_sequence(
        sequence: list[tuple[np.ndarray, np.ndarray]],
        gates: np.ndarray,
        recording_path: str = "rollout.rrd",
) -> None:
    """
    Visualize a rollout using Rerun and write it to `recording_path`.

    Args:
      sequence: list of (x, u) pairs where
        x: (21,) state
        u: (4,) action
      gates: array of gates, shape (N, >=8) with rows
        [gate_id, cx, cy, cz, qw, qx, qy, qz, ...]
      recording_path: output .rrd file
    """
    if len(sequence) == 0:
        raise ValueError("Empty sequence")

    rr.init("flytrack_viz", spawn=False)
    rr.save(recording_path)

    # Static scene: world frame + gates
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    gates = np.asarray(gates, dtype=np.float32)
    if gates.ndim != 2 or gates.shape[1] < 8:
        raise ValueError("gates must have shape (N, >=8): [id, cx, cy, cz, qw, qx, qy, qz, ...]")

    gate_centers = gates[:, 1:4].astype(np.float32)
    gate_quats = gates[:, 4:8].astype(np.float32)

    # Optionally log gate centers
    rr.log("world/gates/centers", rr.Points3D(gate_centers), static=True)

    outer_size = 2.7
    inner_size = 1.5
    for row, center, quat in zip(gates, gate_centers, gate_quats):
        gate_id = int(row[0])
        outer_loop, inner_loop = _make_gate_loops(center, quat, outer_size, inner_size)
        rr.log(
            f"world/gates/gate_{gate_id}/frame",
            rr.LineStrips3D([outer_loop, inner_loop]),
            static=True,
        )

    # TODO: Trajectory visualization
    # TODO:
    #   - Extract positions p_t from each state x_t (hint: p_t = x_t[0:3])
    #   - Build a (T, 3) array and log it as a polyline:
    #       rr.log("world/trajectory", rr.LineStrips3D([positions]), static=True)
    #
    # positions = ...
    # rr.log("world/trajectory", rr.LineStrips3D([positions]), static=True)

    # TODO: Per-step logging (drone + time series)
    #   For each step t:
    #     - set a time axis (sequence index):
    #         rr.set_time("step", sequence=t)
    #     - unpack x into:
    #         pos = x[0:3]
    #         quat = x[9:13]         # [qw,qx,qy,qz]
    #         vel = x[3:6]
    #         body_rates = x[13:16]  # [wx,wy,wz]
    #     - convert quat -> Euler:
    #         euler = quat_to_euler_rpy(quat)
    #     - draw drone model:
    #         motors_world, arms, cam_dir_world = _drone_body_points_world(pos, euler)
    #         rr.log("world/drone/motors", rr.Points3D(...))
    #         rr.log("world/drone/arms", rr.LineStrips3D(...))
    #     - log scalars:
    #         speed = ||vel||, angular_speed = ||body_rates||, and the 4 action components
    #

    raise NotImplementedError("TODO: implement trajectory + per-step logging")
