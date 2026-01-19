import jax
import jax.numpy as jnp


def quat_conjugate(q: jax.Array) -> jax.Array:
    # q = [w, x, y, z]
    return jnp.array([q[0], -q[1], -q[2], -q[3]], dtype=q.dtype)


def quat_mul(q1: jax.Array, q2: jax.Array) -> jax.Array:
    # Hamilton product; q = [w, x, y, z]
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return jnp.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=q1.dtype,
    )


def quat_normalize(q: jax.Array, eps: float = 1e-12) -> jax.Array:
    return q / (jnp.linalg.norm(q) + eps)


def quat_rotate(q: jax.Array, v: jax.Array) -> jax.Array:
    """
    Rotate vector v by quaternion q (assumes q maps body->world).
    """
    vq = jnp.array([0.0, v[0], v[1], v[2]], dtype=q.dtype)
    return quat_mul(quat_mul(q, vq), quat_conjugate(q))[1:4]


def yaw_to_quat(yaw: jax.Array) -> jax.Array:
    half = 0.5 * yaw
    return jnp.array([jnp.cos(half), 0.0, 0.0, jnp.sin(half)], dtype=jnp.float32)


def quat_to_euler(q: jax.Array) -> jax.Array:
    q = quat_normalize(q)
    w, x, y, z = q

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = jnp.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    sinp = jnp.clip(sinp, -1.0, 1.0)
    pitch = jnp.arcsin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = jnp.arctan2(siny_cosp, cosy_cosp)

    return jnp.array([roll, pitch, yaw], dtype=jnp.float32)


def world_to_body(q_body_to_world: jax.Array, v_world: jax.Array) -> jax.Array:
    """
    Rotate world-frame vector into body frame using inverse rotation.
    If q maps body->world, then q* v_body * q_conj = v_world.
    So v_body = q_conj * v_world * q.
    """
    q_inv = quat_conjugate(q_body_to_world)  # unit quaternion inverse
    vq = jnp.array([0.0, v_world[0], v_world[1], v_world[2]], dtype=q_body_to_world.dtype)
    return quat_mul(quat_mul(q_inv, vq), q_body_to_world)[1:4]
