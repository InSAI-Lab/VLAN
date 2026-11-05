from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from vlan_final.eval.ias import binary_score, bounded_error_score, mention_score, weighted_average
from vlan_final.utils.dataset_io import load_frames, target_user_payload
from vlan_final.utils.geometry import obstacle_local_from_frame, quat_to_yaw_deg, world_delta_to_local
from vlan_final.utils.io import load_json, save_json


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def has_stop_intent(text: str) -> bool:
    t = normalize_text(text)
    return bool(re.search(r"\bstop\b", t)) or "do not continue" in t or "blocked" in t or "alternative path" in t


def canonical_obstacle_class(obstacle: dict[str, Any]) -> str:
    subtype = obstacle["semantic_subtype"]
    mapping = {
        "cone": "cone",
        "construction_fence": "construction_fence",
        "traffic_barrier": "traffic_barrier",
        "bench_obstacle": "bench",
        "trash_obstacle": "trash_bin",
        "bicycle_obstacle": "bicycle",
        "scooter_obstacle": "scooter",
        "motorcycle_obstacle": "motorcycle",
        "speed_bump": "speed_bump",
        "stone_cluster": "stone_cluster",
        "stacked_boxes": "low_overhead_box",
        "tree_overhead": "low_tree_branch",
        "tree_branch_or_canopy": "low_tree_branch",
        "street_sign_overhead": "overhead_sign",
        "no_walking": "traffic_sign",
        "warning_bicycle_related": "traffic_sign",
        "surface_drop_or_pit": "drop_off_hazard",
        "stairs_or_steps": "stairs_or_steps",
    }
    return mapping.get(subtype, "unknown")


CANONICAL_ALIASES = {
    "cone": ["traffic cone", "road cone", "orange cone", "cone"],
    "construction_fence": ["construction fence", "fence", "barrier fence"],
    "traffic_barrier": ["traffic barrier", "barrier", "road barrier"],
    "bench": ["bench", "park bench", "seat"],
    "trash_bin": ["trash bin", "trash can", "garbage bin", "bin"],
    "bicycle": ["bicycle", "bike", "cycle"],
    "scooter": ["scooter", "e-scooter", "electric scooter"],
    "motorcycle": ["motorcycle", "motorbike", "bike"],
    "speed_bump": ["speed bump", "bump"],
    "stone_cluster": ["stone cluster", "stones", "rocks", "rock pile"],
    "low_overhead_box": ["low overhead box obstacle", "overhead box", "box obstacle", "boxes overhead", "boxes"],
    "low_tree_branch": ["low tree branch obstacle", "low branch", "tree branch", "branch"],
    "overhead_sign": ["low overhead street sign", "overhead sign", "street sign", "sign overhead"],
    "traffic_sign": ["no-walking sign", "traffic sign", "warning sign", "sign"],
    "drop_off_hazard": ["edge drop-off hazard area", "drop-off", "drop off", "hazard area", "edge hazard"],
    "stairs_or_steps": ["steps", "stairs", "stair area"],
}

GENERIC_OBSTACLE_TERMS = ["obstacle", "object ahead", "something ahead", "hazard", "barrier"]


def extract_predicted_obstacle_class(text: str) -> str | None:
    t = normalize_text(text)
    for canonical, aliases in CANONICAL_ALIASES.items():
        for alias in aliases:
            if alias in t:
                return canonical
    return None


def mentions_obstacle(text: str) -> bool:
    t = normalize_text(text)
    if extract_predicted_obstacle_class(t) is not None:
        return True
    return any(term in t for term in GENERIC_OBSTACLE_TERMS)


def predicted_label_from_instruction(text: str) -> str:
    t = normalize_text(text)
    if has_stop_intent(t):
        return "Blocked"
    if any(k in t for k in ["cautious", "carefully", "slow down", "unstable", "narrow or unstable", "confirming that the route is safe"]):
        return "HighRisk"
    if any(
        k in t
        for k in [
            "go around",
            "around it",
            "bypass",
            "avoid it",
            "avoid the obstacle",
            "avoid the object",
            "avoid this obstacle",
            "move slightly to the left",
            "move slightly to the right",
            "turn left to avoid",
            "turn right to avoid",
            "slightly turn left",
            "slightly turn right",
            "slightly turn to the left",
            "slightly turn to the right",
            "detour",
            "pass it safely",
        ]
    ):
        return "Bypass"
    return "Passable"


def normalize_predicted_label(label: str | None) -> str | None:
    if not label:
        return None
    mapping = {
        "passable": "Passable",
        "bypass": "Bypass",
        "highrisk": "HighRisk",
        "high risk": "HighRisk",
        "blocked": "Blocked",
    }
    return mapping.get(normalize_text(label))


def predicted_side(text: str) -> str:
    t = normalize_text(text)
    left_patterns = [
        r"move slightly to the left",
        r"move to the left",
        r"keep slightly to the left",
        r"keep to the left",
        r"turn left to avoid",
        r"slightly turn left",
        r"slightly turn to the left",
        r"veer left",
        r"go around .* on the left",
        r"bypass .* on the left",
        r"pass .* on the left",
        r"from the left",
    ]
    right_patterns = [
        r"move slightly to the right",
        r"move to the right",
        r"keep slightly to the right",
        r"keep to the right",
        r"turn right to avoid",
        r"slightly turn right",
        r"slightly turn to the right",
        r"veer right",
        r"go around .* on the right",
        r"bypass .* on the right",
        r"pass .* on the right",
        r"from the right",
    ]
    left_match = any(re.search(p, t) for p in left_patterns)
    right_match = any(re.search(p, t) for p in right_patterns)
    if left_match and not right_match:
        return "left"
    if right_match and not left_match:
        return "right"
    return "unspecified"


def has_distance_reference(text: str) -> bool:
    return bool(re.search(r"\b\d+(\.\d+)?\s*(meter|meters|m)\b", normalize_text(text)))


def mentions_goal(text: str) -> bool:
    return "goal" in normalize_text(text)


def blocked_hard_fail_from_instruction(text: str) -> bool:
    t = normalize_text(text)
    if not has_stop_intent(t):
        return True
    positive_forward = (
        re.search(r"(?<!do not )(?<!don't )continue forward", t) is not None
        or re.search(r"(?<!do not )(?<!don't )continue straight", t) is not None
        or "walk forward" in t
        or "move forward" in t
    )
    turn_or_bypass = any(
        k in t
        for k in [
            "turn left to avoid",
            "turn right to avoid",
            "go around",
            "bypass",
            "avoid the obstacle",
            "avoid the object",
            "pass it safely",
        ]
    )
    return positive_forward or turn_or_bypass


def extract_meter_values(text: str) -> list[float]:
    return [float(m.group(1)) for m in re.finditer(r"\b(\d+(?:\.\d+)?)\s*(?:meter|meters|m)\b", normalize_text(text))]


def aligned_distance_errors(gt_text: str, pred_text: str) -> dict[str, Any]:
    gt_vals = extract_meter_values(gt_text)
    pred_vals = extract_meter_values(pred_text)
    pair_count = min(len(gt_vals), len(pred_vals))
    abs_errors = [abs(gt_vals[i] - pred_vals[i]) for i in range(pair_count)]
    return {
        "gt_distance_values_m": gt_vals,
        "pred_distance_values_m": pred_vals,
        "distance_pair_count": pair_count,
        "distance_abs_errors_m": abs_errors,
        "distance_first_abs_error_m": abs_errors[0] if pair_count >= 1 else None,
        "distance_second_abs_error_m": abs_errors[1] if pair_count >= 2 else None,
        "distance_mean_abs_error_m": (sum(abs_errors) / pair_count) if pair_count else None,
    }


def obstacle_radius_m(obstacle: dict[str, Any]) -> float:
    mapping = {
        "cone": 0.25,
        "construction_fence": 0.35,
        "traffic_barrier": 0.35,
        "bench_obstacle": 0.40,
        "trash_obstacle": 0.32,
        "bicycle_obstacle": 0.35,
        "scooter_obstacle": 0.35,
        "motorcycle_obstacle": 0.40,
        "speed_bump": 0.20,
        "stone_cluster": 0.30,
        "stacked_boxes": 0.35,
        "tree_overhead": 0.30,
        "tree_branch_or_canopy": 0.30,
        "street_sign_overhead": 0.28,
        "no_walking": 0.30,
        "warning_bicycle_related": 0.28,
        "surface_drop_or_pit": 0.35,
        "stairs_or_steps": 0.35,
    }
    return mapping.get(obstacle["semantic_subtype"], 0.30)


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _unit_to(src: tuple[float, float], dst: tuple[float, float]) -> tuple[float, float]:
    dx = dst[0] - src[0]
    dz = dst[1] - src[1]
    norm = math.hypot(dx, dz)
    if norm < 1e-8:
        return (0.0, 1.0)
    return (dx / norm, dz / norm)


def _advance(src: tuple[float, float], dst: tuple[float, float], distance: float) -> tuple[float, float]:
    ux, uz = _unit_to(src, dst)
    return (src[0] + ux * distance, src[1] + uz * distance)


def _point_to_segment_distance(p: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    abx = b[0] - a[0]
    abz = b[1] - a[1]
    apx = p[0] - a[0]
    apz = p[1] - a[1]
    denom = abx * abx + abz * abz
    if denom < 1e-8:
        return _dist(p, a)
    t = max(0.0, min(1.0, (apx * abx + apz * abz) / denom))
    proj = (a[0] + t * abx, a[1] + t * abz)
    return _dist(p, proj)


def _polyline_min_obstacle_distance(points: list[tuple[float, float]], obstacle: tuple[float, float]) -> float:
    if len(points) < 2:
        return _dist(points[0], obstacle)
    return min(_point_to_segment_distance(obstacle, points[i], points[i + 1]) for i in range(len(points) - 1))


def _compute_local_goal_and_obstacle(context: dict[str, Any], obstacle: dict[str, Any]) -> tuple[tuple[float, float], tuple[float, float]]:
    start = context["start_frame"]["position_world"]
    goal = context["goal_frame"]["position_world"]
    yaw = quat_to_yaw_deg(context["start_frame"]["rotation_world"])
    goal_dx, goal_dz = world_delta_to_local(
        yaw,
        [float(goal[0]) - float(start[0]), 0.0, float(goal[2]) - float(start[2])],
    )
    obs_local = obstacle_local_from_frame(context["start_frame"], obstacle["position_world"])
    return (goal_dx, goal_dz), (obs_local["dx_local"], obs_local["dz_local"])


def execute_instruction_plan(pred: str, gt_label: str, pred_label: str, pred_side: str, context: dict[str, Any], obstacle: dict[str, Any]) -> dict[str, Any]:
    goal = (0.0, 0.0)
    obs = (0.0, 0.0)
    if gt_label != "Blocked":
        goal, obs = _compute_local_goal_and_obstacle(context, obstacle)
    goal_dist = math.hypot(goal[0], goal[1])
    radius = obstacle_radius_m(obstacle)
    clearance = radius + 0.45
    values = extract_meter_values(pred)
    points: list[tuple[float, float]] = [(0.0, 0.0)]

    if gt_label == "Blocked":
        final = (0.0, 0.0)
        min_obs_dist = _dist(final, obs) if obstacle.get("position_world") else float("inf")
        ok = has_stop_intent(pred) and not blocked_hard_fail_from_instruction(pred)
        return {
            "path_points_local": points,
            "final_local": final,
            "goal_error_l2_xz": 0.0,
            "min_obstacle_distance": min_obs_dist,
            "collision_free": True,
            "pass_side_follow": None,
            "goal_reached_0_3": False,
            "goal_reached_0_5": False,
            "stop_success": ok,
            "ensr_0_3_success": ok,
            "ensr_0_5_success": ok,
        }

    if pred_label == "Bypass":
        if len(values) >= 2:
            approach = values[0]
            remaining = values[1]
        elif len(values) == 1:
            total = values[0]
            approach = max(min(obs[1] - radius - 0.10, total * 0.60), 0.3)
            remaining = max(total - approach, 0.0)
        else:
            total = goal_dist if mentions_goal(pred) else max(goal_dist, 0.8)
            approach = max(min(obs[1] - radius - 0.10, total * 0.60), 0.3)
            remaining = max(total - approach, 0.0)
        approach_z = min(max(approach, 0.3), max(obs[1] - radius - 0.20, 0.3))
        approach_pt = (0.0, approach_z)
        points.append(approach_pt)
        if pred_side == "left":
            bypass_x = obs[0] - clearance
        elif pred_side == "right":
            bypass_x = obs[0] + clearance
        else:
            bypass_x = 0.0
        side_entry_pt = (bypass_x, approach_z + 0.15)
        pass_z = max(obs[1] + radius + 0.20, side_entry_pt[1] + 0.20)
        bypass_pt = (bypass_x, pass_z)
        points.append(side_entry_pt)
        points.append(bypass_pt)
        final = _advance(bypass_pt, goal, min(max(remaining, 0.0), _dist(bypass_pt, goal)))
        points.append(final)
        pass_side_follow = None
        if pred_side == "left":
            pass_side_follow = bypass_pt[0] < obs[0]
        elif pred_side == "right":
            pass_side_follow = bypass_pt[0] > obs[0]
    else:
        travel = values[0] if values else goal_dist
        final = _advance((0.0, 0.0), goal, min(max(travel, 0.0), goal_dist))
        points.append(final)
        pass_side_follow = None

    goal_error = _dist(final, goal)
    min_obs_dist = _polyline_min_obstacle_distance(points, obs)
    collision_free = min_obs_dist > radius
    goal_reached_0_3 = goal_error <= 0.3
    goal_reached_0_5 = goal_error <= 0.5

    if gt_label == "Passable":
        ensr_0_3 = goal_reached_0_3
        ensr_0_5 = goal_reached_0_5
    else:
        ensr_0_3 = bool(pass_side_follow) and collision_free and goal_reached_0_3
        ensr_0_5 = bool(pass_side_follow) and collision_free and goal_reached_0_5

    return {
        "path_points_local": points,
        "final_local": final,
        "goal_error_l2_xz": goal_error,
        "min_obstacle_distance": min_obs_dist,
        "collision_free": collision_free,
        "pass_side_follow": pass_side_follow,
        "goal_reached_0_3": goal_reached_0_3,
        "goal_reached_0_5": goal_reached_0_5,
        "stop_success": False,
        "ensr_0_3_success": ensr_0_3,
        "ensr_0_5_success": ensr_0_5,
    }


def _base_context(scene_dir: Path, traj_name: str) -> dict[str, Any]:
    keyframes = load_frames(scene_dir, traj_name)
    return {
        "start_frame": keyframes[2],
        "goal_frame": keyframes[7],
    }


def _normalize_instruction_prediction(parsed_output: Any) -> dict[str, Any]:
    if isinstance(parsed_output, dict):
        return parsed_output
    if isinstance(parsed_output, str):
        return {
            "predicted_accessibility_label": "",
            "predicted_obstacle_description": "",
            "navigation_instruction": parsed_output,
        }
    return {
        "predicted_accessibility_label": "",
        "predicted_obstacle_description": "",
        "navigation_instruction": "",
    }


def evaluate_traj(
    data_root: Path,
    artifacts_root: Path,
    scene_name: str,
    traj_name: str,
    model_key: str,
    prompt_id: str,
    task_name: str = "humanoid",
) -> dict[str, Any]:
    target = load_json(artifacts_root / "targets" / scene_name / traj_name / "agent_targets.json")
    pred_file = artifacts_root / "predictions" / task_name / model_key / prompt_id / scene_name / f"{traj_name}.json"
    pred_wrap = load_json(pred_file)
    parsed_output = _normalize_instruction_prediction(pred_wrap.get("parsed_output", {}))
    pred = parsed_output.get("navigation_instruction", "")
    humanoid_target = target_user_payload(target, "humanoid")
    gt = humanoid_target["ground_truth_instruction"]
    gt_label = humanoid_target["accessibility_label"]
    context = _base_context(data_root / scene_name, traj_name)
    obstacle = target["obstacle"]

    parse_success = isinstance(pred, str) and len(pred.strip()) > 0
    structured_label = normalize_predicted_label(parsed_output.get("predicted_accessibility_label"))
    pred_label = structured_label if structured_label is not None else (predicted_label_from_instruction(pred) if parse_success else "ParseFailed")
    gt_side = predicted_side(gt)
    pred_side = predicted_side(pred) if parse_success else "unspecified"
    gt_obstacle_class = canonical_obstacle_class(obstacle)
    obstacle_text_for_eval = parsed_output.get("predicted_obstacle_description") or parsed_output.get("predicted_obstacle") or pred
    pred_obstacle_class = extract_predicted_obstacle_class(obstacle_text_for_eval) if parse_success else None
    label_match = pred_label == gt_label
    side_evaluated = gt_label in {"Bypass", "HighRisk"} and gt_side != "unspecified"
    side_match = (pred_side == gt_side) if side_evaluated else None
    obstacle_mentioned = mentions_obstacle(obstacle_text_for_eval) if parse_success else False
    obstacle_semantic_evaluated = pred_obstacle_class is not None
    obstacle_semantic_match = (pred_obstacle_class == gt_obstacle_class) if obstacle_semantic_evaluated else None
    distance_mentioned = has_distance_reference(pred) if parse_success else False
    distance_analysis = aligned_distance_errors(gt, pred) if parse_success else {
        "gt_distance_values_m": [],
        "pred_distance_values_m": [],
        "distance_pair_count": 0,
        "distance_abs_errors_m": [],
        "distance_first_abs_error_m": None,
        "distance_second_abs_error_m": None,
        "distance_mean_abs_error_m": None,
    }
    blocked_hard_fail = gt_label == "Blocked" and blocked_hard_fail_from_instruction(pred)
    passable_over_avoidance = gt_label == "Passable" and pred_label in {"Bypass", "HighRisk"}
    obstacle_ok = obstacle_semantic_match is True
    distance_ok = (
        distance_mentioned
        and distance_analysis["distance_mean_abs_error_m"] is not None
        and distance_analysis["distance_mean_abs_error_m"] <= 0.5
    ) if parse_success else False
    strict_match = (
        parse_success
        and label_match
        and obstacle_ok
        and distance_ok
        and (side_match is True if side_evaluated else True)
        and not blocked_hard_fail
        and not passable_over_avoidance
    )
    ias_components = {
        "label": (0.25, binary_score(label_match)),
        "obstacle": (0.20, mention_score(obstacle_mentioned, obstacle_semantic_match)),
        "direction": (0.20, binary_score(side_match, default_when_none=1.0)),
        "distance": (0.15, bounded_error_score(distance_analysis["distance_mean_abs_error_m"], tolerance=0.5, default_when_missing=0.0)),
        "compliance": (
            0.20,
            0.0
            if blocked_hard_fail or passable_over_avoidance
            else 1.0,
        ),
    }
    ias_score = weighted_average(ias_components)
    success = strict_match
    execution = execute_instruction_plan(pred, gt_label, pred_label, pred_side, context, obstacle) if parse_success else {
        "path_points_local": [(0.0, 0.0)],
        "final_local": (0.0, 0.0),
        "goal_error_l2_xz": float("inf"),
        "min_obstacle_distance": 0.0,
        "collision_free": False,
        "pass_side_follow": None,
        "goal_reached_0_3": False,
        "goal_reached_0_5": False,
        "stop_success": False,
        "ensr_0_3_success": False,
        "ensr_0_5_success": False,
    }
    return {
        "scene_name": scene_name,
        "traj_name": traj_name,
        "ground_truth_label": gt_label,
        "predicted_label": pred_label,
        "parse_success": parse_success,
        "label_match": label_match,
        "ground_truth_side": gt_side,
        "predicted_side": pred_side,
        "side_evaluated": side_evaluated,
        "side_match": side_match,
        "ground_truth_obstacle_class": gt_obstacle_class,
        "predicted_obstacle_class": pred_obstacle_class,
        "obstacle_mentioned": obstacle_mentioned,
        "obstacle_semantic_evaluated": obstacle_semantic_evaluated,
        "obstacle_semantic_match": obstacle_semantic_match,
        "distance_mentioned": distance_mentioned,
        "distance_ok": distance_ok,
        **distance_analysis,
        "ias_score": ias_score,
        "blocked_hard_fail": blocked_hard_fail,
        "passable_over_avoidance": passable_over_avoidance,
        "strict_match": strict_match,
        "success": success,
        "execution_path_points_local": execution["path_points_local"],
        "execution_final_local": execution["final_local"],
        "execution_goal_error_l2_xz": execution["goal_error_l2_xz"],
        "execution_min_obstacle_distance": execution["min_obstacle_distance"],
        "execution_collision_free": execution["collision_free"],
        "execution_pass_side_follow": execution["pass_side_follow"],
        "execution_goal_reached_0_3": execution["goal_reached_0_3"],
        "execution_goal_reached_0_5": execution["goal_reached_0_5"],
        "execution_stop_success": execution["stop_success"],
        "ensr_0_3_success": execution["ensr_0_3_success"],
        "ensr_0_5_success": execution["ensr_0_5_success"],
        "ground_truth_instruction": gt,
        "predicted_instruction": pred,
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(results)
    parse_success = sum(1 for r in results if r["parse_success"])
    label_match = sum(1 for r in results if r["label_match"])
    obstacle_mentioned = sum(1 for r in results if r["obstacle_mentioned"])
    obstacle_semantic_eval = [r for r in results if r["obstacle_semantic_evaluated"]]
    obstacle_semantic_match = sum(1 for r in obstacle_semantic_eval if r["obstacle_semantic_match"])
    distance_mentioned = sum(1 for r in results if r["distance_mentioned"])
    distance_eval = [r for r in results if r["distance_pair_count"] > 0]
    distance_first_eval = [r for r in results if r["distance_first_abs_error_m"] is not None]
    distance_second_eval = [r for r in results if r["distance_second_abs_error_m"] is not None]
    side_eval = [r for r in results if r["side_evaluated"]]
    side_match = sum(1 for r in side_eval if r["side_match"])
    blocked_cases = [r for r in results if r["ground_truth_label"] == "Blocked"]
    blocked_hard_fail = sum(1 for r in blocked_cases if r["blocked_hard_fail"])
    passable_cases = [r for r in results if r["ground_truth_label"] == "Passable"]
    passable_over_avoidance = sum(1 for r in passable_cases if r["passable_over_avoidance"])
    strict_match = sum(1 for r in results if r["strict_match"])
    success = sum(1 for r in results if r["success"])
    ias_mean = sum(r["ias_score"] for r in results) / count if count else 0.0
    goal_reached_0_3 = sum(1 for r in results if r["execution_goal_reached_0_3"])
    goal_reached_0_5 = sum(1 for r in results if r["execution_goal_reached_0_5"])
    collision_free = sum(1 for r in results if r["execution_collision_free"])
    pass_side_exec = [r for r in results if r["execution_pass_side_follow"] is not None]
    pass_side_exec_follow = sum(1 for r in pass_side_exec if r["execution_pass_side_follow"])
    mean_l2_xz = sum(r["execution_goal_error_l2_xz"] for r in results) / count if count else 0.0
    mean_obs_dist = sum(r["execution_min_obstacle_distance"] for r in results) / count if count else 0.0
    ensr_0_3 = sum(1 for r in results if r["ensr_0_3_success"])
    ensr_0_5 = sum(1 for r in results if r["ensr_0_5_success"])
    mean_distance_abs_error = (
        sum(r["distance_mean_abs_error_m"] for r in distance_eval if r["distance_mean_abs_error_m"] is not None) / len(distance_eval)
        if distance_eval
        else None
    )
    mean_distance_first_abs_error = (
        sum(r["distance_first_abs_error_m"] for r in distance_first_eval) / len(distance_first_eval)
        if distance_first_eval
        else None
    )
    mean_distance_second_abs_error = (
        sum(r["distance_second_abs_error_m"] for r in distance_second_eval) / len(distance_second_eval)
        if distance_second_eval
        else None
    )
    gt_counter = Counter(r["ground_truth_label"] for r in results)
    pred_counter = Counter(r["predicted_label"] for r in results)
    return {
        "trajectory_count": count,
        "parse_success_count": parse_success,
        "parse_success_rate": parse_success / count if count else 0.0,
        "label_match_count": label_match,
        "label_match_rate": label_match / count if count else 0.0,
        "obstacle_mentioned_count": obstacle_mentioned,
        "obstacle_mentioned_rate": obstacle_mentioned / count if count else 0.0,
        "obstacle_semantic_evaluated_count": len(obstacle_semantic_eval),
        "obstacle_semantic_match_count": obstacle_semantic_match,
        "obstacle_semantic_match_rate": obstacle_semantic_match / len(obstacle_semantic_eval) if obstacle_semantic_eval else None,
        "distance_mentioned_count": distance_mentioned,
        "distance_mentioned_rate": distance_mentioned / count if count else 0.0,
        "distance_eval_count": len(distance_eval),
        "distance_first_eval_count": len(distance_first_eval),
        "distance_second_eval_count": len(distance_second_eval),
        "distance_mean_abs_error_m": mean_distance_abs_error,
        "distance_first_abs_error_m": mean_distance_first_abs_error,
        "distance_second_abs_error_m": mean_distance_second_abs_error,
        "side_evaluated_count": len(side_eval),
        "side_match_count": side_match,
        "side_match_rate": side_match / len(side_eval) if side_eval else None,
        "blocked_case_count": len(blocked_cases),
        "blocked_hard_fail_count": blocked_hard_fail,
        "blocked_hard_fail_rate": blocked_hard_fail / len(blocked_cases) if blocked_cases else None,
        "passable_case_count": len(passable_cases),
        "passable_over_avoidance_count": passable_over_avoidance,
        "passable_over_avoidance_rate": passable_over_avoidance / len(passable_cases) if passable_cases else None,
        "strict_match_count": strict_match,
        "strict_match_rate": strict_match / count if count else 0.0,
        "success_count": success,
        "success_rate": success / count if count else 0.0,
        "ias_mean": ias_mean,
        "execution_goal_reached_0_3_count": goal_reached_0_3,
        "execution_goal_reached_0_3_rate": goal_reached_0_3 / count if count else 0.0,
        "execution_goal_reached_0_5_count": goal_reached_0_5,
        "execution_goal_reached_0_5_rate": goal_reached_0_5 / count if count else 0.0,
        "execution_collision_free_count": collision_free,
        "execution_collision_free_rate": collision_free / count if count else 0.0,
        "execution_pass_side_follow_count": pass_side_exec_follow,
        "execution_pass_side_follow_rate": pass_side_exec_follow / len(pass_side_exec) if pass_side_exec else None,
        "execution_pass_side_eval_count": len(pass_side_exec),
        "execution_mean_l2_xz_to_goal": mean_l2_xz,
        "execution_mean_obstacle_distance": mean_obs_dist,
        "ensr_0_3_count": ensr_0_3,
        "ensr_0_3_rate": ensr_0_3 / count if count else 0.0,
        "ensr_0_5_count": ensr_0_5,
        "ensr_0_5_rate": ensr_0_5 / count if count else 0.0,
        "ground_truth_label_counts": dict(gt_counter),
        "predicted_label_counts": dict(pred_counter),
    }


def evaluate_dataset(
    data_root: Path,
    artifacts_root: Path,
    model_key: str,
    prompt_id: str,
    scene_names: list[str],
    task_name: str = "humanoid",
) -> dict[str, Any]:
    results = []
    scene_summaries: dict[str, Any] = {}
    eval_root = artifacts_root / "evaluations" / task_name / model_key / prompt_id
    for scene_name in scene_names:
        scene_results = []
        for ctx_path in sorted((artifacts_root / "contexts" / scene_name).glob("*.json")):
            traj_name = ctx_path.stem
            pred_path = artifacts_root / "predictions" / task_name / model_key / prompt_id / scene_name / f"{traj_name}.json"
            if not pred_path.exists():
                continue
            r = evaluate_traj(data_root, artifacts_root, scene_name, traj_name, model_key, prompt_id, task_name=task_name)
            scene_results.append(r)
            results.append(r)
            save_json(eval_root / "per_traj" / scene_name / f"{traj_name}.json", r)
        scene_summary = {"scene_name": scene_name, **summarize(scene_results)}
        scene_summaries[scene_name] = scene_summary
        save_json(eval_root / "per_scene" / f"{scene_name}.json", scene_summary)
    overall = summarize(results)
    summary = {
        "model_key": model_key,
        "version": prompt_id,
        "scene_count": len(scene_names),
        "trajectory_count": len(results),
        "scenes": scene_summaries,
        "overall": overall,
    }
    save_json(eval_root / "summary_all.json", summary)
    return summary
