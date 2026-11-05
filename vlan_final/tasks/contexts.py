from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from vlan_final.targets import bypass_side_from_avoidance_mode, clock_direction_from_local, meters_to_steps, pass_side_clock
from vlan_final.utils.dataset_io import load_frames, target_user_payload
from vlan_final.utils.actions import cumulative_raw, total_planar_distance
from vlan_final.utils.geometry import quat_to_yaw_deg, world_delta_to_local


def _distance_band(distance_m: float) -> str:
    if distance_m < 0.8:
        return "very_near"
    if distance_m < 1.5:
        return "near"
    if distance_m < 2.5:
        return "medium"
    return "far"


def _step_phrase(step_count: int) -> str:
    return f"{step_count} step" if step_count == 1 else f"{step_count} steps"


def _direction_phrase(dx: float, dz: float) -> str:
    if dz >= 0.0:
        if abs(dx) < 0.15:
            return "directly ahead"
        return "ahead on the left" if dx < 0.0 else "ahead on the right"
    if abs(dx) < 0.15:
        return "behind"
    return "behind on the left" if dx < 0.0 else "behind on the right"


def _goal_text(dx_local: float, dz_local: float) -> str:
    if dz_local > 0.15 and abs(dx_local) < 0.08:
        return "directly ahead"
    if dz_local > 0.15 and dx_local < -0.08:
        return "ahead on the left"
    if dz_local > 0.15 and dx_local > 0.08:
        return "ahead on the right"
    if dz_local < -0.15 and abs(dx_local) < 0.08:
        return "behind"
    if dz_local < -0.15 and dx_local < -0.08:
        return "behind on the left"
    if dz_local < -0.15 and dx_local > 0.08:
        return "behind on the right"
    if dx_local < -0.08:
        return "to the left"
    if dx_local > 0.08:
        return "to the right"
    return "near the current position"


def _side_hint_text(side: str, mode: str = "pedestrian") -> str:
    if side not in {"left", "right"}:
        return ""
    if mode == "blind":
        clock = pass_side_clock(side)
        return f" If the user needs to go around it, the safer passing direction is toward {clock}."
    return f" If a bypass is needed, the more accessible passing direction is on the {side} side."


def _blind_risk_hint(subtype: str) -> str:
    mapping = {
        "tree_branch_or_canopy": " It may hang low in the walking line.",
        "stacked_boxes": " It may create a low obstacle in front of the user.",
        "traffic_sign": " It may partially block the walking line.",
        "surface_drop_or_pit": " The ground ahead may drop off or become unsafe.",
        "speed_bump": " The ground surface ahead may change underfoot.",
    }
    return mapping.get(subtype, "")


def _wheelchair_risk_hint(subtype: str) -> str:
    mapping = {
        "tree_branch_or_canopy": " The overhead or side clearance may be limited for a wheelchair.",
        "stacked_boxes": " The route may become narrow for a wheelchair.",
        "traffic_sign": " The forward route may be partially blocked for a wheelchair.",
        "surface_drop_or_pit": " The ground ahead may be unsafe for a wheelchair.",
        "speed_bump": " The surface ahead may be uneven for a wheelchair.",
        "traffic_barrier": " The route may require a narrow detour.",
        "construction_fence": " The route may require a narrow detour.",
        "trash_obstacle": " Clearance around it may be limited.",
        "bench_obstacle": " Clearance around it may be limited.",
    }
    return mapping.get(subtype, "")


def _prefix_motion_summary(actions_payload: dict[str, Any]) -> str:
    prefix_actions = actions_payload["base_actions"]["observed_prefix"]
    total_dx, total_dz, total_yaw = cumulative_raw(prefix_actions)
    distance_m = total_planar_distance(prefix_actions)
    if abs(total_dx) < 0.08:
        lateral = "stayed mostly straight"
    elif total_dx < 0.0:
        lateral = "shifted slightly left"
    else:
        lateral = "shifted slightly right"
    if abs(total_yaw) < 0.5:
        heading = "with almost no heading change"
    elif total_yaw < 0.0:
        heading = "while turning a little to the left"
    else:
        heading = "while turning a little to the right"
    return (
        f"In the first two observed steps, the humanoid moved forward about {distance_m:.1f} meters, "
        f"{lateral}, {heading}."
    )


def _observed_raw_stats(observed_actions: list[dict[str, Any]]) -> dict[str, float]:
    raw = [a["raw"] for a in observed_actions]
    return {
        "avg_abs_dx": sum(abs(float(a["dx"])) for a in raw) / len(raw),
        "avg_dz": sum(float(a["dz"]) for a in raw) / len(raw),
        "avg_abs_yaw": sum(abs(float(a["yaw"])) for a in raw) / len(raw),
        "max_abs_dx": max(abs(float(a["dx"])) for a in raw),
        "max_abs_yaw": max(abs(float(a["yaw"])) for a in raw),
    }


def _sign_hint(value: float, threshold: float = 1e-6) -> str:
    if value > threshold:
        return "positive"
    if value < -threshold:
        return "negative"
    return "near_zero"


def build_context(scene_dir: Path, traj_name: str, targets: dict[str, Any]) -> dict[str, Any]:
    frames = load_frames(scene_dir, traj_name)
    input_frames = frames[:3]
    goal_frame = frames[7]
    obstacle = targets["obstacle"]
    start_frame = frames[2]

    # Humanoid summaries
    start = start_frame["position_world"]
    goal = goal_frame["position_world"]
    yaw = quat_to_yaw_deg(start_frame["rotation_world"])
    goal_dx, goal_dz = world_delta_to_local(yaw, [float(goal[0]) - float(start[0]), 0.0, float(goal[2]) - float(start[2])])
    goal_distance = math.hypot(goal_dx, goal_dz)
    rep = targets["representative_frames"][0]["obstacle_local"]
    obstacle_distance = math.hypot(rep["dx_local"], rep["dz_local"])
    safer_side = bypass_side_from_avoidance_mode(obstacle["avoidance_mode"])
    safer_clock = pass_side_clock(safer_side)
    humanoid_user = target_user_payload(targets, "humanoid")
    blind_user = target_user_payload(targets, "blind")
    wheelchair_user = target_user_payload(targets, "wheelchair")
    ped_label = humanoid_user["accessibility_label"]
    blind_label = blind_user["accessibility_label"]
    wheelchair_label = wheelchair_user["accessibility_label"]

    # Drone and quadruped geometry
    def action_context_for(user_key: str) -> dict[str, Any]:
        user = targets["users"][user_key]
        start_pose = user["rollout_start_pose_world"]
        start_pos = start_pose["position_world"]
        start_yaw_deg = float(start_pose["yaw_world_deg"])
        goal_dx_world = float(goal[0]) - float(start_pos[0])
        goal_dz_world = float(goal[2]) - float(start_pos[2])
        obs_pos = obstacle["position_world"]
        obstacle_dx_world = float(obs_pos[0]) - float(start_pos[0])
        obstacle_dz_world = float(obs_pos[2]) - float(start_pos[2])
        goal_dx_local, goal_dz_local = world_delta_to_local(start_yaw_deg, [goal_dx_world, 0.0, goal_dz_world])
        obstacle_dx_local, obstacle_dz_local = world_delta_to_local(start_yaw_deg, [obstacle_dx_world, 0.0, obstacle_dz_world])
        goal_distance_local = math.hypot(goal_dx_local, goal_dz_local)
        obstacle_distance_local = math.hypot(obstacle_dx_local, obstacle_dz_local)
        required_pass_side = "either"
        if obstacle["avoidance_mode"] == "Right":
            required_pass_side = "left_of_obstacle"
        elif obstacle["avoidance_mode"] == "Left":
            required_pass_side = "right_of_obstacle"
        protected_radius = 0.30
        safety_margin = 0.05
        if required_pass_side == "left_of_obstacle":
            suggested = obstacle_dx_local - (protected_radius + safety_margin)
        elif required_pass_side == "right_of_obstacle":
            suggested = obstacle_dx_local + (protected_radius + safety_margin)
        else:
            suggested = obstacle_dx_local
        return {
            "accessibility_label": user["accessibility_label"],
            "observed_actions": user["observed_prefix_actions"],
            "observed_raw_action_stats": _observed_raw_stats(user["observed_prefix_actions"]),
            "rollout_start_pose_world": start_pose,
            "goal_world_absolute": {
                "frame_order": goal_frame["order"],
                "frame_idx": goal_frame["frame_idx"],
                "position_world": goal_frame["position_world"],
            },
            "goal_world_relative_to_rollout_start": {"dx_world": goal_dx_world, "dz_world": goal_dz_world},
            "goal_local_relative_to_rollout_start": {"dx_local": goal_dx_local, "dz_local": goal_dz_local},
            "goal_distance_m": goal_distance_local,
            "goal_position_text": _goal_text(goal_dx_local, goal_dz_local),
            "goal_distance_band": _distance_band(goal_distance_local),
            "obstacle_world_relative_to_rollout_start": {"dx_world": obstacle_dx_world, "dz_world": obstacle_dz_world},
            "obstacle_local_relative_to_rollout_start": {"dx_local": obstacle_dx_local, "dz_local": obstacle_dz_local},
            "obstacle_distance_m": obstacle_distance_local,
            "obstacle_position_text": _goal_text(obstacle_dx_local, obstacle_dz_local),
            "obstacle_distance_band": _distance_band(obstacle_distance_local),
            "required_pass_side": required_pass_side,
            "protected_radius_hint": protected_radius,
            "suggested_bypass_corridor_dx_local": suggested,
            "early_bypass_direction_hint": _sign_hint(suggested),
            "goal_lateral_direction_hint": _sign_hint(goal_dx_local),
            "naive_goal_per_step_local": {"dx_local_per_step": goal_dx_local / 5.0, "dz_local_per_step": goal_dz_local / 5.0},
        }

    return {
        "scene_name": scene_dir.name,
        "traj_name": traj_name,
        "images": [{"order": int(f["order"]), "img_name": f["img_name"], "rgb_path": f["rgb_path"]} for f in input_frames],
        "obstacle": obstacle,
        "corrected_accessibility": targets["corrected_accessibility"],
        "users": {
            "humanoid": {
                "accessibility_label": ped_label,
                "target_instruction": humanoid_user["ground_truth_instruction"],
                "prefix_motion_summary": _prefix_motion_summary(targets),
                "goal_summary": (
                    f"The goal is {_direction_phrase(goal_dx, goal_dz)} at a {_distance_band(goal_distance)} distance "
                    f"(about {goal_distance:.1f} meters away)."
                ),
                "obstacle_summary": (
                    f"There is a relevant obstacle {_direction_phrase(rep['dx_local'], rep['dz_local'])} "
                    f"at a {_distance_band(obstacle_distance)} distance (about {obstacle_distance:.1f} meters). "
                    f"If a detour is needed, the safer route is on the {safer_side} side of that obstacle."
                ),
            },
            "blind": {
                "target_instruction": blind_user["ground_truth_instruction"],
                "goal_clock_direction": clock_direction_from_local(goal_dx, goal_dz),
                "goal_step_count": meters_to_steps(goal_distance),
                "obstacle_clock_direction": clock_direction_from_local(rep["dx_local"], rep["dz_local"]),
                "obstacle_step_count": meters_to_steps(obstacle_distance),
                "safer_pass_clock_direction": safer_clock,
                "prefix_motion_summary": (
                    f"In the first two observed steps, the user moved about {_step_phrase(meters_to_steps(total_planar_distance(targets['base_actions']['observed_prefix'])))} "
                    f"and stayed generally aligned with 12 o'clock."
                ),
                "goal_summary": (
                    f"The goal is around {clock_direction_from_local(goal_dx, goal_dz)} "
                    f"at about {_step_phrase(meters_to_steps(goal_distance))} away."
                ),
                "obstacle_summary": (
                    f"There is a relevant obstacle around {clock_direction_from_local(rep['dx_local'], rep['dz_local'])} "
                    f"at about {_step_phrase(meters_to_steps(obstacle_distance))} away."
                    f"{_blind_risk_hint(obstacle['semantic_subtype'])}"
                    f"{_side_hint_text(safer_side if blind_label != 'Blocked' else '', mode='blind')}"
                ),
            },
            "wheelchair": {
                "target_instruction": wheelchair_user["ground_truth_instruction"],
                "goal_summary": (
                    f"The goal is {_direction_phrase(goal_dx, goal_dz)} at a {_distance_band(goal_distance)} distance "
                    f"(about {goal_distance:.1f} meters away)."
                ),
                "obstacle_summary": (
                    f"There is a relevant obstacle {_direction_phrase(rep['dx_local'], rep['dz_local'])} "
                    f"at a {_distance_band(obstacle_distance)} distance (about {obstacle_distance:.1f} meters away)."
                    f"{_wheelchair_risk_hint(obstacle['semantic_subtype'])}"
                    f"{_side_hint_text(safer_side if wheelchair_label != 'Blocked' else '', mode='wheelchair')}"
                ),
                "prefix_motion_summary": _prefix_motion_summary(targets),
            },
            "drone": action_context_for("drone"),
            "quadrupedRobot": action_context_for("quadrupedRobot"),
        },
    }
