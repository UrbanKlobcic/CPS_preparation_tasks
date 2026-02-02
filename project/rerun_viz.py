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

import argparse
import numpy as np
import rerun as rr


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
        return np.array(
            [
                [-h, 0.0, -h],
                [h, 0.0, -h],
                [h, 0.0, h],
                [-h, 0.0, h],
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


_FRONT_SPAN = 0.18  
_BACK_SPAN = 0.28 
_FB_DIST = 0.30 

_front_x = +_FB_DIST / 2.0
_back_x = -_FB_DIST / 2.0
_front_y_off = _FRONT_SPAN / 2.0
_back_y_off = _BACK_SPAN / 2.0


_DRONE_MOTORS_BODY = np.array(
    [
        [_front_x, +_front_y_off, 0.0],  
        [_front_x, -_front_y_off, 0.0], 
        [_back_x, +_back_y_off, 0.0], 
        [_back_x, -_back_y_off, 0.0], 
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

def _log_scalar(path: str, value: float) -> None:
    """Compat shim: log a single scalar using rr.Scalars."""
    rr.log(path, rr.Scalars(np.array([value], dtype=np.float32)))


def _load_rollout_as_sequence(npy_path: str) -> list[tuple[np.ndarray, np.ndarray]]:
    data = np.load(npy_path, allow_pickle=True)

    if isinstance(data, list) or (isinstance(data, np.ndarray) and data.shape == ()):
        if isinstance(data, np.ndarray):
            data = data.item()
        
        if not isinstance(data, list):
             raise ValueError(f"Loaded data is {type(data)}, expected list or numpy array.")

        seq = []
        for i, item in enumerate(data):
            # item should be tuple (x, u)
            x, u = np.asarray(item[0], dtype=np.float32), np.asarray(item[1], dtype=np.float32)
            x = x.reshape(-1)
            u = u.reshape(-1)
            seq.append((x, u))
        return seq

    # Handle numpy array format (N, 2) objects or structured
    if isinstance(data, np.ndarray) and data.dtype == object:
        if data.ndim == 2 and data.shape[1] == 2:
            seq = []
            for t in range(data.shape[0]):
                x = np.asarray(data[t, 0], dtype=np.float32).reshape(-1)
                u = np.asarray(data[t, 1], dtype=np.float32).reshape(-1)
                seq.append((x, u))
            return seq
        
        # data shape (T,) containing tuples
        if data.ndim == 1:
            seq = []
            for i, item in enumerate(data):
                x, u = np.asarray(item[0], dtype=np.float32), np.asarray(item[1], dtype=np.float32)
                seq.append((x.reshape(-1), u.reshape(-1)))
            return seq

    # Case B: numeric matrix rollout (T, D)
    if data.ndim != 2:
        raise ValueError(f"Expected 2D rollout array or list, got shape {data.shape}")

    T, D = data.shape

    if D == 25:
        xs = data[:, :21].astype(np.float32)
        us = data[:, 21:25].astype(np.float32)
        return [(xs[t], us[t]) for t in range(T)]

    if D == 22:
        xs18 = data[:, :18].astype(np.float32)
        us = data[:, 18:22].astype(np.float32)

        seq = []
        for t in range(T):
            s = xs18[t]
            x = np.zeros((21,), dtype=np.float32)

            x[0:3] = s[0:3]
            x[3:6] = s[3:6]
            x[6:9] = s[6:9]
            x[9:13] = s[9:13]
            x[13:16] = s[13:16]

            prev_like = float(s[16])
            batt_like = float(s[17])
            if -1.1 <= prev_like <= 1.1:
                x[19] = prev_like  
            if 20.0 <= batt_like <= 30.0:
                x[20] = batt_like  
            else:
                x[20] = 24.0

            seq.append((x, us[t]))
        return seq

    raise ValueError(
        f"Unsupported rollout shape (T, D)=({T}, {D}). "
        "Expected list, D=25 ([21|4]) or D=22 ([18|4])."
    )

def visualize_state_action_sequence(
        sequence: list[tuple[np.ndarray, np.ndarray]],
        gates: np.ndarray,
        recording_path: str = "rollout.rrd",
        app_id: str = "flytrack_viz",
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

    rr.init(app_id, spawn=False)
    rr.save(recording_path)

    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    gates = np.asarray(gates, dtype=np.float32)
    if gates.ndim != 2 or gates.shape[1] < 8:
        raise ValueError("gates must have shape (N, >=8): [id, cx, cy, cz, qw, qx, qy, qz, ...]")

    gate_centers = gates[:, 1:4].astype(np.float32)
    gate_quats = gates[:, 4:8].astype(np.float32)

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
    
    # Static trajectory polyline
    positions = np.stack([np.asarray(x, dtype=np.float32)[0:3] for (x, _) in sequence], axis=0)
    rr.log("world/trajectory", rr.LineStrips3D([positions]), static=True) #if we want "growing" trajectory, we would log it per step without static=True

    # Per-step logging: drone + plots
    dt = 0.01  
    cam_ray_len = 2.0  

    for t, (x, u) in enumerate(sequence):
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        u = np.asarray(u, dtype=np.float32).reshape(-1)

        if x.shape[0] != 21:
            raise ValueError(f"State at t={t} has shape {x.shape}, expected (21,)")
        if u.shape[0] != 4:
            raise ValueError(f"Action at t={t} has shape {u.shape}, expected (4,)")

        rr.set_time("step", sequence=t)
        rr.set_time("sim_time", duration=t * dt)

        pos = x[0:3]
        vel = x[3:6]
        quat = x[9:13]          # [qw,qx,qy,qz]
        body_rates = x[13:16]   # [wx,wy,wz]

        # quat -> Euler (roll,pitch,yaw)
        euler = quat_to_euler_rpy(quat)

        # Drone geometry in world
        motors_world, arms, cam_dir_world = _drone_body_points_world(pos, euler)

        # Drone visuals
        rr.log("world/drone/center", rr.Points3D([pos]))
        rr.log("world/drone/motors", rr.Points3D(motors_world))
        rr.log("world/drone/arms", rr.LineStrips3D(arms))

        # Camera ray
        cam_end = pos + cam_dir_world * cam_ray_len
        cam_ray = np.stack([pos, cam_end], axis=0).astype(np.float32)
        rr.log("world/drone/camera_ray", rr.LineStrips3D([cam_ray]))

        speed = float(np.linalg.norm(vel))
        ang_speed = float(np.linalg.norm(body_rates))

        _log_scalar("plots/speed", speed)
        _log_scalar("plots/angular_speed", ang_speed)

        _log_scalar("plots/actions/u_roll", float(u[0]))
        _log_scalar("plots/actions/u_pitch", float(u[1]))
        _log_scalar("plots/actions/u_yaw", float(u[2]))
        _log_scalar("plots/actions/u_thrust", float(u[3]))

        _log_scalar("plots/orientation/roll", float(euler[0]))
        _log_scalar("plots/orientation/pitch", float(euler[1]))
        _log_scalar("plots/orientation/yaw", float(euler[2]))
    

if __name__ == "__main__":
    gates_np  = np.array(
    [
        [0, 12.500000, 2.000000, 1.350000, -0.707107, 0.000000, 0.000000, 0.707107],
        [1, 6.500000, 6.000000, 1.350000, -0.382684, 0.000000, 0.000000, 0.923879], 
        [2, 5.500000, 14.000000, 1.350000, -0.258819, 0.000000, 0.000000, 0.965926],
        [3, 2.500000, 24.000000, 1.350000, 0.000000, 0.000000, 0.000000, 1.000000], 
        [4, 7.500000, 30.000000, 1.350000, -0.642788, 0.000000, 0.000000, 0.766044],
        [8, 18.500000, 22.000000, 1.350000, -0.087155, 0.000000, 0.000000, 0.996195]
        [9, 20.500000, 14.000000, 1.350000, 0.087155, 0.000000, 0.000000, 0.996195],
        [10, 18.500000, 6.000000, 1.350000, 0.382684, 0.000000, 0.000000, 0.923879],
    ], dtype=np.float32,)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rollout",
        type=str,
        default="flytrack_eval_trajectory_0.npy",
        help="Path to rollout .npy (e.g. example rollout.npy).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="rollout.rrd",
        help="Output .rrd file path.",
    )
    args = parser.parse_args()

    seq = _load_rollout_as_sequence(args.rollout)
    visualize_state_action_sequence(seq, gates_np, recording_path=args.out)
    print(f"Wrote: {args.out}")
