from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any

from vlan_final.utils.dataset_io import load_frames
from vlan_final.utils.geometry import quat_to_yaw_deg, wrap_angle_deg
from vlan_final.utils.io import load_json, save_json


def l2_xz(a: list[float], b: list[float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[2]) - float(b[2]))


def rollout_from_pose(start_pose: dict[str, Any], actions: list[dict[str, Any]]) -> dict[str, Any]:
    pos = list(start_pose["position_world"])
    yaw = float(start_pose["yaw_world_deg"])
    path_positions = [pos.copy()]
    for action in actions:
        dx = float(action["dx"])
        dz = float(action["dz"])
        dyaw = float(action["yaw"])
        theta = math.radians(yaw)
        right = [math.cos(theta), 0.0, math.sin(theta)]
        forward = [math.sin(theta), 0.0, -math.cos(theta)]
        pos[0] += dx * right[0] + dz * forward[0]
        pos[2] += dx * right[2] + dz * forward[2]
        yaw = wrap_angle_deg(yaw + dyaw)
        path_positions.append(pos.copy())
    return {"final_position": pos, "final_yaw_deg": yaw, "path_positions_world": path_positions}


def point_to_segment_distance_xz(p: list[float], a: list[float], b: list[float]) -> float:
    px, pz = float(p[0]), float(p[2])
    ax, az = float(a[0]), float(a[2])
    bx, bz = float(b[0]), float(b[2])
    abx = bx - ax
    abz = bz - az
    apx = px - ax
    apz = pz - az
    ab_len_sq = abx * abx + abz * abz
    if ab_len_sq < 1e-12:
        return math.hypot(px - ax, pz - az)
    t = max(0.0, min(1.0, (apx * abx + apz * abz) / ab_len_sq))
    cx = ax + t * abx
    cz = az + t * abz
    return math.hypot(px - cx, pz - cz)


def collision_check(path_positions: list[list[float]], obstacle_position: list[float], collision_radius: float) -> dict[str, Any]:
    min_dist = float("inf")
    collided = False
    collided_segment = None
    for idx in range(len(path_positions) - 1):
        dist = point_to_segment_distance_xz(obstacle_position, path_positions[idx], path_positions[idx + 1])
        if dist < min_dist:
            min_dist = dist
        if dist <= collision_radius and not collided:
            collided = True
            collided_segment = idx
    return {
        "collision": collided,
        "collision_radius": collision_radius,
        "min_distance_to_obstacle_xz": min_dist,
        "collided_segment": collided_segment,
    }


def mean_abs_action(actions: list[dict[str, Any]], key: str) -> float:
    return sum(abs(float(a[key])) for a in actions) / len(actions) if actions else 0.0


def max_abs_action(actions: list[dict[str, Any]], key: str) -> float:
    return max(abs(float(a[key])) for a in actions) if actions else 0.0


def _action_component(action: dict[str, Any], key: str) -> float:
    if key in action:
        return float(action[key])
    raw = action.get("raw")
    if isinstance(raw, dict) and key in raw:
        return float(raw[key])
    raise KeyError(key)


def path_length_from_actions(actions: list[dict[str, Any]]) -> float:
    total = 0.0
    for action in actions:
        dx = _action_component(action, "dx")
        dz = _action_component(action, "dz")
        total += math.hypot(dx, dz)
    return total


def spl_score(success: bool, predicted_length: float, optimal_length: float) -> float:
    if not success:
        return 0.0
    if optimal_length <= 1e-9:
        return 1.0 if predicted_length <= 1e-9 else 0.0
    return optimal_length / max(predicted_length, optimal_length)


def _path_side_follow(start_pose: dict[str, Any], path_positions: list[list[float]], obstacle_position: list[float], required_pass_side: str) -> bool | None:
    if required_pass_side not in {"left_of_obstacle", "right_of_obstacle"}:
        return None
    theta = math.radians(float(start_pose["yaw_world_deg"]))
    right = [math.cos(theta), 0.0, math.sin(theta)]
    start = start_pose["position_world"]
    obs_rel_x = (float(obstacle_position[0]) - float(start[0])) * right[0] + (float(obstacle_position[2]) - float(start[2])) * right[2]
    obs_rel_z = (float(obstacle_position[0]) - float(start[0])) * math.sin(theta) + (float(obstacle_position[2]) - float(start[2])) * (-math.cos(theta))
    if len(path_positions) < 2:
        return None
    best_x = None
    best_depth = float("inf")
    for p in path_positions:
        rel_x = (float(p[0]) - float(start[0])) * right[0] + (float(p[2]) - float(start[2])) * right[2]
        rel_z = (float(p[0]) - float(start[0])) * math.sin(theta) + (float(p[2]) - float(start[2])) * (-math.cos(theta))
        depth_gap = abs(rel_z - obs_rel_z)
        if depth_gap < best_depth:
            best_depth = depth_gap
            best_x = rel_x
    if best_x is None:
        return None
    if required_pass_side == "left_of_obstacle":
        return best_x < obs_rel_x
    return best_x > obs_rel_x


def _normalize_action_prediction(parsed: Any) -> list[dict[str, Any]]:
    if isinstance(parsed, dict) and isinstance(parsed.get("predicted_actions"), list):
        return parsed["predicted_actions"]
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict) and all(k in parsed for k in ("dx", "dz", "yaw")):
        return [parsed]
    raise KeyError("predicted_actions")


def evaluate_traj(
    artifacts_root: Path,
    data_root: Path,
    scene_name: str,
    traj_name: str,
    task: str,
    model_key: str,
    prompt_id: str,
    goal_tolerance: float = 0.30,
    collision_radius: float = 0.30,
    stop_displacement_tolerance: float = 0.05,
    stop_yaw_tolerance_deg: float = 5.0,
) -> dict[str, Any]:
    pred_wrap = load_json(artifacts_root / "predictions" / task / model_key / prompt_id / scene_name / f"{traj_name}.json")
    parsed = pred_wrap["parsed_output"]
    target = load_json(artifacts_root / "targets" / scene_name / traj_name / "agent_targets.json")
    context = load_json(artifacts_root / "contexts" / scene_name / f"{traj_name}.json")
    user = target["users"][task]
    label = str(user["accessibility_label"])
    predicted_actions = _normalize_action_prediction(parsed)
    gt_actions = user["ground_truth_actions"]
    start_pose = context["users"][task]["rollout_start_pose_world"]
    goal_position = context["users"][task]["goal_world_absolute"]["position_world"]
    goal_yaw_deg = quat_to_yaw_deg(load_frames(data_root / scene_name, traj_name)[7]["rotation_world"])
    obstacle_position = target["obstacle"]["position_world"]
    rollout = rollout_from_pose(start_pose, predicted_actions)
    final_position = rollout["final_position"]
    final_yaw_deg = rollout["final_yaw_deg"]
    goal_error = l2_xz(final_position, goal_position)
    goal_reached = goal_error <= goal_tolerance
    yaw_error_deg = abs(wrap_angle_deg(final_yaw_deg - goal_yaw_deg))
    start_displacement = l2_xz(final_position, start_pose["position_world"])
    start_yaw_deg = float(start_pose["yaw_world_deg"])
    stop_yaw_change_deg = abs(wrap_angle_deg(final_yaw_deg - start_yaw_deg))
    path_positions = rollout["path_positions_world"]
    collision = collision_check(path_positions, obstacle_position, collision_radius)
    collision_free = not collision["collision"]
    collision_considered = label in {"Bypass", "HighRisk"}
    stop_success = (
        start_displacement <= stop_displacement_tolerance
        and stop_yaw_change_deg <= stop_yaw_tolerance_deg
        and max_abs_action(predicted_actions, "dz") <= stop_displacement_tolerance
        and max_abs_action(predicted_actions, "dx") <= stop_displacement_tolerance
    )
    if label == "Passable":
        success = goal_reached
        success_reason = "goal_reached_only"
    elif label in {"Bypass", "HighRisk"}:
        success = goal_reached and collision_free
        success_reason = "goal_reached_and_collision_free"
    elif label == "Blocked":
        success = stop_success
        success_reason = "blocked_stop_success"
    else:
        success = goal_reached
        success_reason = "fallback_goal_reached_only"

    predicted_path_length = path_length_from_actions(predicted_actions)
    gt_path_length = path_length_from_actions(gt_actions)
    if label == "Blocked":
        spl = 1.0 if stop_success else 0.0
        safe_spl = spl
    else:
        safe_success = goal_reached and collision_free
        spl = spl_score(success, predicted_path_length, gt_path_length)
        safe_spl = spl_score(safe_success, predicted_path_length, gt_path_length)
    pass_side_follow = _path_side_follow(start_pose, path_positions, obstacle_position, context["users"][task]["required_pass_side"])
    return {
        "scene_name": scene_name,
        "traj_name": traj_name,
        "model_key": model_key,
        "accessibility_label": label,
        "generation_method": user.get("generation_method"),
        "goal_tolerance": goal_tolerance,
        "collision_radius": collision_radius,
        "stop_displacement_tolerance": stop_displacement_tolerance,
        "stop_yaw_tolerance_deg": stop_yaw_tolerance_deg,
        "predicted_final_pose_world": {"position_world": final_position, "yaw_world_deg": final_yaw_deg},
        "start_pose_world": {"position_world": start_pose["position_world"], "yaw_world_deg": start_yaw_deg},
        "ground_truth_goal_pose_world": {"position_world": goal_position, "yaw_world_deg": goal_yaw_deg},
        "error": {
            "l2_xz_to_goal": goal_error,
            "yaw_err_deg_to_goal": yaw_error_deg,
            "l2_xz_from_start": start_displacement,
            "yaw_change_deg_from_start": stop_yaw_change_deg,
        },
        "path_behavior": {
            "predicted_path_length": predicted_path_length,
            "ground_truth_path_length": gt_path_length,
            "mean_abs_dx": mean_abs_action(predicted_actions, "dx"),
            "mean_abs_dz": mean_abs_action(predicted_actions, "dz"),
            "mean_abs_yaw": mean_abs_action(predicted_actions, "yaw"),
            "max_abs_dx": max_abs_action(predicted_actions, "dx"),
            "max_abs_dz": max_abs_action(predicted_actions, "dz"),
            "max_abs_yaw": max_abs_action(predicted_actions, "yaw"),
        },
        "collision_check": collision,
        "collision_considered_for_success": collision_considered,
        "goal_reached": goal_reached,
        "collision_free": collision_free,
        "stop_success": stop_success,
        "pass_side_follow": pass_side_follow,
        "success": success,
        "spl": spl,
        "safe_spl": safe_spl,
        "success_reason": success_reason,
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    by_label = {label: [r for r in results if r["accessibility_label"] == label] for label in ["Passable", "Bypass", "HighRisk", "Blocked"]}
    label_counts = Counter(r["accessibility_label"] for r in results)
    method_counts = Counter(str(r.get("generation_method", "")) for r in results)

    def summarize_group(items: list[dict[str, Any]]) -> dict[str, Any]:
        if not items:
            return {"count": 0}
        count = len(items)
        sum_goal = sum(r["error"]["l2_xz_to_goal"] for r in items)
        sum_yaw = sum(r["error"]["yaw_err_deg_to_goal"] for r in items)
        goal_reached = sum(1 for r in items if r["goal_reached"])
        collision_free = sum(1 for r in items if r["collision_free"])
        stop_success = sum(1 for r in items if r["stop_success"])
        success = sum(1 for r in items if r["success"])
        spl_sum = sum(float(r.get("spl", 0.0)) for r in items)
        safe_spl_sum = sum(float(r.get("safe_spl", 0.0)) for r in items)
        return {
            "count": count,
            "goal_reached_count": goal_reached,
            "collision_free_count": collision_free,
            "stop_success_count": stop_success,
            "success_count": success,
            "goal_reached_rate": goal_reached / count,
            "collision_free_rate": collision_free / count,
            "stop_success_rate": stop_success / count,
            "success_rate": success / count,
            "spl": spl_sum / count,
            "safe_spl": safe_spl_sum / count,
            "mean_l2_xz_to_goal": sum_goal / count,
            "mean_yaw_err_deg_to_goal": sum_yaw / count,
        }

    return {
        "trajectory_count": total,
        "label_counts": dict(label_counts),
        "generation_method_counts": dict(method_counts),
        "overall": summarize_group(results),
        "per_label": {label: summarize_group(items) for label, items in by_label.items()},
    }


def tolerance_report(results: list[dict[str, Any]], model_key: str) -> dict[str, Any]:
    label_counts = Counter(r["accessibility_label"] for r in results)
    out = {"model_key": model_key, "trajectory_count": len(results), "label_counts": dict(label_counts), "goal_tolerance_reports": {}}
    for tol in [0.3, 0.5]:
        adjusted = []
        for r in results:
            r2 = dict(r)
            goal_reached = r["error"]["l2_xz_to_goal"] <= tol
            if r["accessibility_label"] == "Passable":
                success = goal_reached
            elif r["accessibility_label"] in {"Bypass", "HighRisk"}:
                success = goal_reached and r["collision_free"]
            elif r["accessibility_label"] == "Blocked":
                success = r["stop_success"]
            else:
                success = goal_reached
            if r["accessibility_label"] == "Blocked":
                spl = 1.0 if r["stop_success"] else 0.0
                safe_spl = spl
            else:
                pred_len = float(r["path_behavior"]["predicted_path_length"])
                gt_len = float(r["path_behavior"]["ground_truth_path_length"])
                safe_success = goal_reached and r["collision_free"]
                spl = spl_score(success, pred_len, gt_len)
                safe_spl = spl_score(safe_success, pred_len, gt_len)
            r2["_goal_reached_tol"] = goal_reached
            r2["_success_tol"] = success
            r2["_spl_tol"] = spl
            r2["_safe_spl_tol"] = safe_spl
            adjusted.append(r2)
        count = len(adjusted)
        side_eval = [r for r in adjusted if r["pass_side_follow"] is not None]
        out["goal_tolerance_reports"][str(tol)] = {
            "success_rate": sum(1 for r in adjusted if r["_success_tol"]) / count if count else 0.0,
            "goal_reached_rate": sum(1 for r in adjusted if r["_goal_reached_tol"]) / count if count else 0.0,
            "collision_free_rate": sum(1 for r in adjusted if r["collision_free"]) / count if count else 0.0,
            "spl": sum(r["_spl_tol"] for r in adjusted) / count if count else 0.0,
            "safe_spl": sum(r["_safe_spl_tol"] for r in adjusted) / count if count else 0.0,
            "mean_l2_xz_to_goal": sum(r["error"]["l2_xz_to_goal"] for r in adjusted) / count if count else 0.0,
            "mean_obs_dist": sum(r["collision_check"]["min_distance_to_obstacle_xz"] for r in adjusted) / count if count else 0.0,
            "pass_side_follow_rate": (sum(1 for r in side_eval if r["pass_side_follow"]) / len(side_eval)) if side_eval else None,
            "pass_side_eval_count": len(side_eval),
        }
    return out


def evaluate_dataset(artifacts_root: Path, data_root: Path, task: str, model_key: str, prompt_id: str, scene_names: list[str]) -> dict[str, Any]:
    results = []
    scene_summaries = {}
    eval_root = artifacts_root / "evaluations" / task / model_key / prompt_id
    for scene_name in scene_names:
        scene_results = []
        for ctx_path in sorted((artifacts_root / "contexts" / scene_name).glob("*.json")):
            traj_name = ctx_path.stem
            pred_path = artifacts_root / "predictions" / task / model_key / prompt_id / scene_name / f"{traj_name}.json"
            if not pred_path.exists():
                continue
            r = evaluate_traj(artifacts_root, data_root, scene_name, traj_name, task, model_key, prompt_id)
            scene_results.append(r)
            results.append(r)
            save_json(eval_root / "per_traj" / scene_name / f"{traj_name}.json", r)
        scene_summary = {"scene_name": scene_name, **summarize(scene_results)}
        scene_summaries[scene_name] = scene_summary
        save_json(eval_root / "per_scene" / f"{scene_name}.json", scene_summary)
    overall = summarize(results)
    summary = {"model_key": model_key, "scene_count": len(scene_names), "trajectory_count": len(results), "scenes": scene_summaries, "overall": overall}
    save_json(eval_root / "summary_all.json", summary)
    save_json(eval_root / "tolerance_report.json", tolerance_report(results, model_key))
    return summary
