from __future__ import annotations

import math
from typing import Any

from .geometry import dot, normalize, quat_rotate_vector, quat_to_yaw_deg, wrap_angle_deg, world_delta_to_local


def clean_round(value: float, digits: int = 2, eps: float = 1e-8) -> float:
    v = round(float(value), digits)
    if abs(v) < eps:
        return 0.0
    return v


def compute_actions_from_keyframes(
    keyframes: dict[str, Any],
    fps: float = 15.0,
    target_speed: float = 1.0,
    round_digits: int = 2,
) -> dict[str, Any]:
    frames = keyframes.get("frames", [])
    if len(frames) < 2:
        return {"actions": []}

    requested_step = keyframes.get("requested_step", 15)
    dt = requested_step / fps
    actions = []
    for i in range(len(frames) - 1):
        frame0 = frames[i]
        frame1 = frames[i + 1]
        p0 = frame0["position_world"]
        p1 = frame1["position_world"]
        q0 = frame0["rotation_world"]
        q1 = frame1["rotation_world"]
        delta_world = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]]

        right = quat_rotate_vector(q0, [1.0, 0.0, 0.0])
        forward = quat_rotate_vector(q0, [0.0, 0.0, 1.0])
        right[1] = 0.0
        forward[1] = 0.0
        right = normalize(right)
        forward = normalize(forward)

        dx = dot(delta_world, right)
        dz = dot(delta_world, forward)
        yaw0 = quat_to_yaw_deg(q0)
        yaw1 = quat_to_yaw_deg(q1)
        dyaw = wrap_angle_deg(yaw1 - yaw0)
        distance = math.sqrt(dx * dx + dz * dz)
        speed = distance / dt if dt > 0 else 0.0
        scale = target_speed / speed if speed > 1e-8 else 1.0
        actions.append(
            {
                "id": i,
                "from_order": frame0["order"],
                "to_order": frame1["order"],
                "from_frame_idx": frame0["frame_idx"],
                "to_frame_idx": frame1["frame_idx"],
                "dt": clean_round(dt, round_digits),
                "distance": clean_round(distance, round_digits),
                "speed": clean_round(speed, round_digits),
                "scale": clean_round(scale, round_digits),
                "raw": {
                    "dx": clean_round(dx, round_digits),
                    "dz": clean_round(dz, round_digits),
                    "yaw": clean_round(dyaw, round_digits),
                },
                "normalized": {
                    "target_speed": clean_round(target_speed, round_digits),
                    "dx": clean_round(dx * scale, round_digits),
                    "dz": clean_round(dz * scale, round_digits),
                    "yaw": clean_round(dyaw * scale, round_digits),
                },
            }
        )
    return {
        "trajectory_name": keyframes.get("trajectory_name"),
        "requested_step": keyframes.get("requested_step"),
        "requested_front_count": keyframes.get("requested_front_count"),
        "requested_center_count": keyframes.get("requested_center_count"),
        "requested_back_count": keyframes.get("requested_back_count"),
        "num_keyframes": keyframes.get("num_keyframes", len(frames)),
        "selected_frame_indices": keyframes.get("selected_frame_indices", []),
        "middle_frame_idx": keyframes.get("middle_frame_idx"),
        "middle_frame_order": keyframes.get("middle_frame_order"),
        "fps": fps,
        "target_speed": target_speed,
        "num_actions": len(actions),
        "actions": actions,
    }


def cumulative_raw(actions: list[dict[str, Any]]) -> tuple[float, float, float]:
    return (
        sum(float(a["raw"]["dx"]) for a in actions),
        sum(float(a["raw"]["dz"]) for a in actions),
        sum(float(a["raw"]["yaw"]) for a in actions),
    )


def total_planar_distance(actions: list[dict[str, Any]]) -> float:
    return sum(math.hypot(float(a["raw"]["dx"]), float(a["raw"]["dz"])) for a in actions)


def rollout_pose_from_frame(frame: dict[str, Any], actions: list[dict[str, Any]]) -> dict[str, Any]:
    pos = list(frame["position_world"])
    yaw = quat_to_yaw_deg(frame["rotation_world"])
    for action in actions:
        dx = float(action["raw"]["dx"])
        dz = float(action["raw"]["dz"])
        dyaw = float(action["raw"]["yaw"])
        theta = math.radians(yaw)
        right = [math.cos(theta), 0.0, math.sin(theta)]
        forward = [math.sin(theta), 0.0, -math.cos(theta)]
        pos[0] += dx * right[0] + dz * forward[0]
        pos[2] += dx * right[2] + dz * forward[2]
        yaw = wrap_angle_deg(yaw + dyaw)
    return {"position_world": pos, "yaw_world_deg": yaw}


def slice_future_actions(actions_payload: dict[str, Any], start_order: int = 2, end_order: int = 7) -> list[dict[str, Any]]:
    actions = actions_payload.get("actions", [])
    return [a for a in actions if a["from_order"] >= start_order and a["to_order"] <= end_order]


def interpolate_polyline(points: list[tuple[float, float]], num_segments: int) -> list[tuple[float, float]]:
    seg_lengths = []
    total = 0.0
    for i in range(len(points) - 1):
        dx = points[i + 1][0] - points[i][0]
        dz = points[i + 1][1] - points[i][1]
        length = math.hypot(dx, dz)
        seg_lengths.append(length)
        total += length
    if total < 1e-9:
        return [points[0] for _ in range(num_segments + 1)]
    samples = [points[0]]
    for k in range(1, num_segments):
        target = total * (k / num_segments)
        acc = 0.0
        for idx, length in enumerate(seg_lengths):
            next_acc = acc + length
            if target <= next_acc or idx == len(seg_lengths) - 1:
                t = 0.0 if length < 1e-9 else (target - acc) / length
                x = points[idx][0] + t * (points[idx + 1][0] - points[idx][0])
                z = points[idx][1] + t * (points[idx + 1][1] - points[idx][1])
                samples.append((x, z))
                break
            acc = next_acc
    samples.append(points[-1])
    return samples


def plan_drone_full_straight_actions(frames: list[dict[str, Any]], all_actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    start_frame = frames[0]
    goal_frame = frames[-1]
    start_pos = list(start_frame["position_world"])
    goal_pos = list(goal_frame["position_world"])
    start_yaw = quat_to_yaw_deg(start_frame["rotation_world"])
    goal_yaw = quat_to_yaw_deg(goal_frame["rotation_world"])
    num_steps = len(all_actions)
    waypoints = []
    for i in range(num_steps + 1):
        t = i / num_steps
        waypoints.append([
            start_pos[0] * (1.0 - t) + goal_pos[0] * t,
            start_pos[1] * (1.0 - t) + goal_pos[1] * t,
            start_pos[2] * (1.0 - t) + goal_pos[2] * t,
        ])
    current_yaw = start_yaw
    planned = []
    for i in range(num_steps):
        p0 = waypoints[i]
        p1 = waypoints[i + 1]
        dx_local, dz_local = world_delta_to_local(current_yaw, [p1[0] - p0[0], 0.0, p1[2] - p0[2]])
        remaining = num_steps - i
        yaw_step = wrap_angle_deg(goal_yaw - current_yaw) / remaining
        planned.append(
            {
                "id": all_actions[i]["id"],
                "from_order": all_actions[i]["from_order"],
                "to_order": all_actions[i]["to_order"],
                "generation_method": "drone_full_straight_line_plan",
                "raw": {"dx": round(dx_local, 2), "dz": round(dz_local, 2), "yaw": round(yaw_step, 2)},
            }
        )
        current_yaw = wrap_angle_deg(current_yaw + yaw_step)
    return planned


def stop_actions_like(base_actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": a["id"],
            "from_order": a["from_order"],
            "to_order": a["to_order"],
            "generation_method": "blocked_stop_sequence",
            "raw": {"dx": 0.0, "dz": 0.0, "yaw": 0.0},
        }
        for a in base_actions
    ]


def plan_drone_overhead_bypass_actions(
    start_frame: dict[str, Any],
    goal_frame: dict[str, Any],
    obstacle_world: list[float],
    base_actions: list[dict[str, Any]],
    pass_side: str,
    clearance_radius: float = 0.40,
) -> list[dict[str, Any]]:
    obstacle_x, obstacle_z = world_delta_to_local(
        quat_to_yaw_deg(start_frame["rotation_world"]),
        [
            float(obstacle_world[0]) - float(start_frame["position_world"][0]),
            0.0,
            float(obstacle_world[2]) - float(start_frame["position_world"][2]),
        ],
    )
    goal_x, goal_z = world_delta_to_local(
        quat_to_yaw_deg(start_frame["rotation_world"]),
        [
            float(goal_frame["position_world"][0]) - float(start_frame["position_world"][0]),
            0.0,
            float(goal_frame["position_world"][2]) - float(start_frame["position_world"][2]),
        ],
    )
    if pass_side == "left":
        corridor_x = min(goal_x, obstacle_x - clearance_radius)
    elif pass_side == "right":
        corridor_x = max(goal_x, obstacle_x + clearance_radius)
    else:
        corridor_x = goal_x
    pre_z = max(0.25, obstacle_z - 0.20)
    post_z = min(goal_z - 0.20, obstacle_z + 0.30) if goal_z > obstacle_z + 0.20 else obstacle_z + 0.20
    if post_z <= pre_z:
        post_z = pre_z + 0.20
    polyline = [(0.0, 0.0), (corridor_x, pre_z), (corridor_x, post_z), (goal_x, goal_z)]
    samples = interpolate_polyline(polyline, num_segments=len(base_actions))
    actions = []
    for idx, base in enumerate(base_actions):
        x0, z0 = samples[idx]
        x1, z1 = samples[idx + 1]
        actions.append(
            {
                "id": base["id"],
                "from_order": base["from_order"],
                "to_order": base["to_order"],
                "generation_method": "drone_overhead_bypass_path",
                "raw": {"dx": round(x1 - x0, 2), "dz": round(z1 - z0, 2), "yaw": 0.0},
            }
        )
    return actions

