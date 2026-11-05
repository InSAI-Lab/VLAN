from __future__ import annotations

import math
from typing import Any


def normalize(v: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in v))
    if norm < 1e-12:
        return [0.0, 0.0, 0.0]
    return [x / norm for x in v]


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def quat_multiply(q1: list[float], q2: list[float]) -> list[float]:
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return [
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ]


def quat_conjugate(q: list[float]) -> list[float]:
    x, y, z, w = q
    return [-x, -y, -z, w]


def quat_rotate_vector(q: list[float], v: list[float]) -> list[float]:
    qv = [v[0], v[1], v[2], 0.0]
    return quat_multiply(quat_multiply(q, qv), quat_conjugate(q))[:3]


def quat_to_yaw_deg(q: list[float]) -> float:
    x, y, z, w = q
    siny_cosp = 2.0 * (w * y + x * z)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw_rad = math.atan2(siny_cosp, cosy_cosp)
    return math.degrees(yaw_rad)


def wrap_angle_deg(angle: float) -> float:
    while angle > 180.0:
        angle -= 360.0
    while angle < -180.0:
        angle += 360.0
    return angle


def yaw_to_forward_right_working(yaw_deg: float) -> tuple[list[float], list[float]]:
    theta = math.radians(yaw_deg)
    right = [math.cos(theta), 0.0, math.sin(theta)]
    forward = [math.sin(theta), 0.0, -math.cos(theta)]
    return forward, right


def world_delta_to_local(yaw_deg: float, delta_world: list[float]) -> tuple[float, float]:
    forward, right = yaw_to_forward_right_working(yaw_deg)
    dx = delta_world[0] * right[0] + delta_world[2] * right[2]
    dz = delta_world[0] * forward[0] + delta_world[2] * forward[2]
    return dx, dz


def obstacle_local_from_frame(frame: dict[str, Any], obstacle_position: list[float]) -> dict[str, float]:
    pos = frame["position_world"]
    yaw = quat_to_yaw_deg(frame["rotation_world"])
    vx = float(obstacle_position[0]) - float(pos[0])
    vz = float(obstacle_position[2]) - float(pos[2])
    forward, right = yaw_to_forward_right_working(yaw)
    dx_local = vx * right[0] + vz * right[2]
    dz_local = vx * forward[0] + vz * forward[2]
    return {"dx_local": dx_local, "dz_local": dz_local}

