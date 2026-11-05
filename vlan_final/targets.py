from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .utils.actions import (
    compute_actions_from_keyframes,
    cumulative_raw,
    plan_drone_full_straight_actions,
    plan_drone_overhead_bypass_actions,
    rollout_pose_from_frame,
    slice_future_actions,
    stop_actions_like,
    total_planar_distance,
)
from .utils.geometry import obstacle_local_from_frame, quat_to_yaw_deg, world_delta_to_local
from .utils.dataset_io import load_accessibility_labels, load_frames, load_keyframes_payload, load_released_target, load_trajectory
from .utils.io import load_json

STEP_LENGTH_M = 0.75


def infer_semantic_subtype(object_id: str) -> str:
    oid = object_id.lower()
    if "cone" in oid:
        return "cone"
    if "fence" in oid:
        return "construction_fence"
    if "barrier" in oid:
        return "traffic_barrier"
    if "bench" in oid:
        return "bench_obstacle"
    if "trash" in oid:
        return "trash_obstacle"
    if "bicy" in oid:
        return "bicycle_obstacle"
    if "scooter" in oid:
        return "scooter_obstacle"
    if "motor" in oid:
        return "motorcycle_obstacle"
    if "speed" in oid and "bump" in oid:
        return "speed_bump"
    if "tree" in oid or "branch" in oid:
        return "tree_branch_or_canopy"
    if "sign" in oid:
        return "traffic_sign"
    if "drop" in oid or "pit" in oid:
        return "surface_drop_or_pit"
    if "box" in oid or "crate" in oid:
        return "stacked_boxes"
    return "unknown"


def pretty_object_name(object_id: str) -> str:
    return " ".join(p for p in object_id.replace("_", " ").split("-") if p)


def bypass_side_from_avoidance_mode(avoidance_mode: str | None) -> str:
    if avoidance_mode == "Right":
        return "left"
    if avoidance_mode == "Left":
        return "right"
    return "either"


def meters_to_steps(distance_m: float) -> int:
    return max(1, int(round(max(distance_m, 0.0) / STEP_LENGTH_M)))


def clock_direction_from_local(dx_local: float, dz_local: float) -> str:
    angle_deg = math.degrees(math.atan2(dx_local, dz_local))
    sector = int(round(angle_deg / 30.0))
    hour_map = {
        0: 12,
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        6: 6,
        -1: 11,
        -2: 10,
        -3: 9,
        -4: 8,
        -5: 7,
        -6: 6,
    }
    sector = max(-6, min(6, sector))
    return f"{hour_map[sector]} o'clock"


def pass_side_clock(side: str) -> str:
    if side == "left":
        return "10 o'clock"
    if side == "right":
        return "2 o'clock"
    return "12 o'clock"


def blind_risk_detail(subtype: str) -> str:
    mapping = {
        "speed_bump": "The ground surface may change underfoot.",
        "tree_branch_or_canopy": "There may be a low branch at head or shoulder height.",
        "stacked_boxes": "There may be a low obstacle projecting into the path.",
        "traffic_sign": "A sign may partially block the route ahead.",
        "surface_drop_or_pit": "The ground ahead may drop off and is unsafe.",
    }
    return mapping.get(subtype, "The path ahead may be risky.")


def select_primary_obstacle(obstacles: list[dict[str, Any]]) -> dict[str, Any]:
    active = [o for o in obstacles if o.get("isGameObjectActive", True)]
    avoidance = [o for o in active if o.get("participateInAvoidance")]
    return avoidance[0] if avoidance else active[0]


def representative_frames(frames: list[dict[str, Any]], obstacle_position: list[float]) -> list[dict[str, Any]]:
    rows = []
    for frame in frames:
        local = obstacle_local_from_frame(frame, obstacle_position)
        rows.append(
            {
                "order": int(frame["order"]),
                "img_name": frame["img_name"],
                "rgb_path": frame["rgb_path"],
                "obstacle_local": local,
                "depth_score": abs(local["dz_local"]),
            }
        )
    rows.sort(key=lambda row: (row["depth_score"], row["order"]))
    best = rows[0]
    chosen = {best["order"]: best}
    prev_order = max(best["order"] - 1, 0)
    for row in rows:
        if row["order"] == prev_order:
            chosen[row["order"]] = row
            break
    return [chosen[k] for k in sorted(chosen)]


def pedestrian_obstacle_name(object_id: str, subtype: str) -> str:
    subtype_map = {
        "cone": "traffic cone",
        "construction_fence": "construction fence",
        "traffic_barrier": "traffic barrier",
        "bench_obstacle": "bench",
        "trash_obstacle": "trash bin",
        "bicycle_obstacle": "bicycle",
        "scooter_obstacle": "scooter",
        "motorcycle_obstacle": "motorcycle",
        "speed_bump": "speed bump",
        "stacked_boxes": "low overhead box obstacle",
        "tree_branch_or_canopy": "low tree branch obstacle",
        "traffic_sign": "traffic sign",
        "surface_drop_or_pit": "edge drop-off hazard area",
    }
    return subtype_map.get(subtype, pretty_object_name(object_id))


def pedestrian_instruction(obstacle: dict[str, Any], base_future_actions: list[dict[str, Any]], label: str) -> str:
    object_id = obstacle["object_id"]
    subtype = obstacle["semantic_subtype"]
    obstacle_name = pedestrian_obstacle_name(object_id, subtype)
    distance_m = total_planar_distance(base_future_actions)
    total_dx, _, _ = cumulative_raw(base_future_actions)
    side = "left" if total_dx < -0.05 else "right" if total_dx > 0.05 else ("left" if bypass_side_from_avoidance_mode(obstacle["avoidance_mode"]) == "left" else "right")
    approach_m = max(0.8, min(1.5, distance_m * 0.45))
    finish_m = max(0.6, distance_m - approach_m)
    if label == "Passable":
        return (
            f"Continue forward along the current path for about {distance_m:.1f} meters. "
            f"The {obstacle_name} ahead does not require a dedicated detour, so keep moving mostly straight, "
            f"make a slight adjustment as you approach it, and then continue toward the goal."
        )
    if label == "Bypass":
        return (
            f"First walk forward along the current route for about {approach_m:.1f} meters. "
            f"As you approach the {obstacle_name} ahead, move slightly to the {side} to go around it, "
            f"then return to your original direction and continue toward the goal for about {finish_m:.1f} more meters."
        )
    if label == "HighRisk":
        return (
            f"Move forward cautiously for about {approach_m:.1f} meters. "
            f"The area around the {obstacle_name} ahead is narrow or unstable, so slow down, keep to the {side}, "
            f"and only continue toward the goal after confirming that the route is safe."
        )
    return f"Do not continue forward. The route ahead is blocked by the {obstacle_name}, so stop and choose a different passable route."


def blind_accessibility_label(ped_label: str, obstacle: dict[str, Any]) -> str:
    subtype = obstacle["semantic_subtype"]
    if ped_label == "Blocked":
        return "Blocked"
    if subtype == "surface_drop_or_pit":
        return "Blocked"
    if subtype in {"speed_bump", "tree_branch_or_canopy", "stacked_boxes", "traffic_sign"}:
        return "HighRisk"
    return ped_label


def wheelchair_accessibility_label(ped_label: str, obstacle: dict[str, Any]) -> str:
    subtype = obstacle["semantic_subtype"]
    if ped_label == "Blocked":
        return "Blocked"
    if subtype in {"surface_drop_or_pit", "stairs_or_steps"}:
        return "Blocked"
    if subtype == "speed_bump":
        return "HighRisk"
    if subtype == "stacked_boxes":
        return "HighRisk" if ped_label in {"Passable", "Bypass"} else ped_label
    if subtype == "tree_branch_or_canopy":
        return "HighRisk" if ped_label == "Bypass" else ped_label
    return ped_label


def blind_instruction(obstacle: dict[str, Any], base_future_actions: list[dict[str, Any]], label: str) -> str:
    obstacle_name = pedestrian_obstacle_name(obstacle["object_id"], obstacle["semantic_subtype"])
    risk_detail = blind_risk_detail(obstacle["semantic_subtype"])
    distance_m = total_planar_distance(base_future_actions)
    total_dx, total_dz, _ = cumulative_raw(base_future_actions)
    side = "left" if total_dx < -0.05 else "right" if total_dx > 0.05 else bypass_side_from_avoidance_mode(obstacle["avoidance_mode"])
    goal_clock = clock_direction_from_local(total_dx, total_dz if abs(total_dz) > 1e-6 else distance_m)
    pass_clock = pass_side_clock(side)
    total_steps = meters_to_steps(distance_m)
    approach_steps = max(1, int(round(total_steps * 0.45)))
    finish_steps = max(1, total_steps - approach_steps)
    if label == "Passable":
        return (
            f"Walk about {total_steps} steps toward {goal_clock}. "
            f"The route stays mostly clear, so keep moving mostly straight and continue carefully."
        )
    if label == "Bypass":
        return (
            f"Walk about {approach_steps} steps toward 12 o'clock. "
            f"There is a {obstacle_name} ahead. Keep slightly toward {pass_clock} to go around it safely. "
            f"After passing it, continue about {finish_steps} more steps toward {goal_clock}."
        )
    if label == "HighRisk":
        return (
            f"Walk about {approach_steps} careful steps toward 12 o'clock. "
            f"There is a risky {obstacle_name} ahead. {risk_detail} "
            f"Slow down, use your cane to check the area carefully, and only edge toward {pass_clock} if the path is clear. "
            f"If it remains safe, continue about {finish_steps} more steps toward {goal_clock}."
        )
    return (
        f"Stop. The route ahead at 12 o'clock is blocked by a {obstacle_name}. "
        f"Do not continue forward. Ask for another safe route."
    )


def wheelchair_instruction(obstacle: dict[str, Any], base_future_actions: list[dict[str, Any]], label: str) -> str:
    obstacle_name = pedestrian_obstacle_name(obstacle["object_id"], obstacle["semantic_subtype"])
    distance_m = total_planar_distance(base_future_actions)
    total_dx, _, _ = cumulative_raw(base_future_actions)
    side = "left" if total_dx < -0.05 else "right" if total_dx > 0.05 else ("left" if bypass_side_from_avoidance_mode(obstacle["avoidance_mode"]) == "left" else "right")
    approach_m = max(0.8, min(1.5, distance_m * 0.45))
    finish_m = max(0.6, distance_m - approach_m)
    if label == "Passable":
        return (
            f"Roll forward for about {distance_m:.1f} meters toward the goal. "
            f"The {obstacle_name} ahead remains passable for a wheelchair, so keep the chair mostly straight and make only a small adjustment if needed."
        )
    if label == "Bypass":
        return (
            f"Roll forward for about {approach_m:.1f} meters. "
            f"As you approach the {obstacle_name}, keep slightly to the {side} to go around it while maintaining clear wheelchair space. "
            f"After passing it, continue toward the goal for about {finish_m:.1f} more meters."
        )
    if label == "HighRisk":
        return (
            f"Move forward cautiously for about {approach_m:.1f} meters. "
            f"The {obstacle_name} ahead makes wheelchair travel narrow, uneven, or unstable, so slow down, keep to the {side}, and continue only if the surface and clearance remain safe. "
            f"After that, continue about {finish_m:.1f} more meters toward the goal if possible."
        )
    return (
        f"Do not continue forward. The route ahead is not safely passable for a wheelchair because of the {obstacle_name}, so stop and choose another accessible route."
    )


def build_agent_targets(scene_dir: Path, traj_name: str) -> dict[str, Any]:
    released = load_released_target(scene_dir, traj_name)
    if released is not None:
        return released

    keyframes = load_keyframes_payload(scene_dir, traj_name)
    trajectory = load_trajectory(scene_dir, traj_name)
    per_user = load_accessibility_labels(scene_dir, traj_name)
    actions_payload = compute_actions_from_keyframes(keyframes)
    frames = load_frames(scene_dir, traj_name)
    obstacle_raw = select_primary_obstacle(trajectory["obstacles"])
    obstacle = {
        "object_id": obstacle_raw.get("objectId"),
        "display_name": obstacle_raw.get("displayName"),
        "position_world": obstacle_raw.get("position"),
        "avoidance_mode": obstacle_raw.get("avoidanceMode"),
        "participate_in_avoidance": obstacle_raw.get("participateInAvoidance"),
        "semantic_subtype": infer_semantic_subtype(obstacle_raw.get("objectId", "")),
    }
    reps = representative_frames(frames, obstacle["position_world"])
    start_frame = frames[2]
    goal_frame = frames[-1]
    base_future_actions = slice_future_actions(actions_payload, start_order=2, end_order=len(frames) - 1)
    quad_label = per_user["quadrupedRobot"]
    quad_full_actions = actions_payload["actions"]
    quad_prefix = quad_full_actions[:2]

    drone_label = per_user["drone"]
    if drone_label == "Passable":
        drone_full_actions = plan_drone_full_straight_actions(frames, actions_payload["actions"])
        drone_prefix = drone_full_actions[:2]
        drone_target = drone_full_actions[2:]
        drone_rollout = rollout_pose_from_frame(frames[0], drone_prefix)
        drone_method = "drone_full_straight_line_plan"
    elif drone_label == "Bypass" and obstacle["semantic_subtype"] in {"stacked_boxes", "tree_branch_or_canopy"}:
        drone_target = plan_drone_overhead_bypass_actions(
            start_frame=frames[2],
            goal_frame=goal_frame,
            obstacle_world=obstacle["position_world"],
            base_actions=base_future_actions,
            pass_side=bypass_side_from_avoidance_mode(obstacle["avoidance_mode"]),
        )
        drone_full_actions = actions_payload["actions"][:2] + drone_target
        drone_prefix = drone_full_actions[:2]
        drone_rollout = {"position_world": start_frame["position_world"], "yaw_world_deg": quat_to_yaw_deg(start_frame["rotation_world"])}
        drone_method = "drone_overhead_bypass_path"
    elif drone_label == "Blocked":
        drone_target = stop_actions_like(base_future_actions)
        drone_full_actions = actions_payload["actions"][:2] + drone_target
        drone_prefix = drone_full_actions[:2]
        drone_rollout = {"position_world": start_frame["position_world"], "yaw_world_deg": quat_to_yaw_deg(start_frame["rotation_world"])}
        drone_method = "blocked_stop_sequence"
    else:
        drone_target = base_future_actions
        drone_full_actions = actions_payload["actions"][:2] + drone_target
        drone_prefix = drone_full_actions[:2]
        drone_rollout = {"position_world": start_frame["position_world"], "yaw_world_deg": quat_to_yaw_deg(start_frame["rotation_world"])}
        drone_method = "base_actions_copy"

    ped_label = per_user["humanoid"]
    blind_label = blind_accessibility_label(ped_label, obstacle)
    wheelchair_label = wheelchair_accessibility_label(ped_label, obstacle)
    uses_new_layout = (scene_dir / traj_name / "metadata.json").exists()
    raw_source = (
        {
            "trajectory_path": f"data/test/{scene_dir.name}/{traj_name}/trajectory.json",
            "metadata_path": f"data/test/{scene_dir.name}/{traj_name}/metadata.json",
        }
        if uses_new_layout
        else {
            "trajectory_path": f"data/test/{scene_dir.name}/Trajectory/{traj_name}.json",
            "keyframes_path": f"data/test/{scene_dir.name}/KeyFrames/{traj_name}/keyframes.json",
            "corrected_accessibility_path": f"data/test/{scene_dir.name}/KeyFrames/{traj_name}/accessibility_manual_v2_4label.json",
        }
    )
    return {
        "scene_name": scene_dir.name,
        "traj_name": traj_name,
        "raw_source": raw_source,
        "obstacle": obstacle,
        "representative_frames": reps,
        "goal": {"frame_order": goal_frame["order"], "position_world": goal_frame["position_world"]},
        "base_actions": {
            "observed_prefix": actions_payload["actions"][:2],
            "full_actions": actions_payload["actions"],
            "future_reference": base_future_actions,
        },
        "corrected_accessibility": {
            "humanoid": ped_label,
            "quadrupedRobot": quad_label,
            "drone": drone_label,
            "blind": blind_label,
            "wheelchair": wheelchair_label,
        },
        "users": {
            "humanoid": {
                "accessibility_label": ped_label,
                "pass_side": bypass_side_from_avoidance_mode(obstacle["avoidance_mode"]),
                "ground_truth_instruction": pedestrian_instruction(obstacle, base_future_actions, ped_label),
                "prompt_id": "pedestrian_v3_6_frozen",
            },
            "blind": {
                "accessibility_label": blind_label,
                "pass_side": bypass_side_from_avoidance_mode(obstacle["avoidance_mode"]),
                "ground_truth_instruction": blind_instruction(obstacle, base_future_actions, blind_label),
                "prompt_id": "blind_instruction_main",
            },
            "wheelchair": {
                "accessibility_label": wheelchair_label,
                "pass_side": bypass_side_from_avoidance_mode(obstacle["avoidance_mode"]),
                "ground_truth_instruction": wheelchair_instruction(obstacle, base_future_actions, wheelchair_label),
                "prompt_id": "wheelchair_instruction_main",
            },
            "quadrupedRobot": {
                "accessibility_label": quad_label,
                "observed_prefix_actions": quad_prefix,
                "rollout_start_pose_world": {
                    "position_world": start_frame["position_world"],
                    "yaw_world_deg": quat_to_yaw_deg(start_frame["rotation_world"]),
                },
                "ground_truth_actions": base_future_actions,
                "prompt_id": "quadruped_action_main",
            },
            "drone": {
                "accessibility_label": drone_label,
                "generation_method": drone_method,
                "observed_prefix_actions": drone_prefix,
                "rollout_start_pose_world": drone_rollout,
                "ground_truth_actions": drone_target,
                "prompt_id": "drone_action_main",
            },
        },
    }
