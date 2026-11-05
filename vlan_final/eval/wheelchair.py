from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from vlan_final.eval.ias import binary_score, bounded_error_score, mention_score, weighted_average
from vlan_final.eval.pedestrian import (
    aligned_distance_errors,
    blocked_hard_fail_from_instruction,
    canonical_obstacle_class,
    extract_predicted_obstacle_class,
    has_distance_reference,
    has_stop_intent,
    mentions_obstacle,
    normalize_predicted_label,
    predicted_label_from_instruction,
    predicted_side,
)
from vlan_final.utils.io import load_json, save_json


def wheelchair_highrisk_compliant(text: str) -> bool:
    t = text.lower()
    keywords = ["cautious", "careful", "slow down", "narrow", "clearance", "surface", "stable", "accessible", "uneven", "safe"]
    return any(k in t for k in keywords)


def wheelchair_passable_ok(pred_text: str, pred_label: str) -> bool:
    t = pred_text.lower()
    if pred_label in {"Bypass", "HighRisk", "Blocked"}:
        return False
    bad = ["stop", "choose another route", "not safely passable", "narrow route", "detour", "go around"]
    return not any(k in t for k in bad)


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


def evaluate_traj(artifacts_root: Path, scene_name: str, traj_name: str, model_key: str, prompt_id: str) -> dict[str, Any]:
    target = load_json(artifacts_root / "targets" / scene_name / traj_name / "agent_targets.json")
    pred_file = artifacts_root / "predictions" / "wheelchair" / model_key / prompt_id / scene_name / f"{traj_name}.json"
    pred_wrap = load_json(pred_file)
    parsed_output = _normalize_instruction_prediction(pred_wrap.get("parsed_output", {}))
    pred = parsed_output.get("navigation_instruction", "")
    gt = target["users"]["wheelchair"]["ground_truth_instruction"]
    gt_label = target["users"]["wheelchair"]["accessibility_label"]
    gt_side = target["users"]["wheelchair"]["pass_side"]
    obstacle = target["obstacle"]

    parse_success = isinstance(pred, str) and len(pred.strip()) > 0
    structured_label = normalize_predicted_label(parsed_output.get("predicted_accessibility_label"))
    pred_label = structured_label if structured_label is not None else (predicted_label_from_instruction(pred) if parse_success else "ParseFailed")
    pred_side = predicted_side(pred) if parse_success else "unspecified"
    side_evaluated = gt_label in {"Bypass", "HighRisk"} and gt_side in {"left", "right"}
    side_match = (pred_side == gt_side) if side_evaluated else None
    gt_obstacle_class = canonical_obstacle_class(obstacle)
    obstacle_text_for_eval = parsed_output.get("predicted_obstacle_description") or pred
    pred_obstacle_class = extract_predicted_obstacle_class(obstacle_text_for_eval) if parse_success else None
    label_match = pred_label == gt_label
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
    blocked_ok = has_stop_intent(pred) and not blocked_hard_fail_from_instruction(pred) if gt_label == "Blocked" else None
    highrisk_ok = wheelchair_highrisk_compliant(pred) if gt_label == "HighRisk" else None
    passable_ok = wheelchair_passable_ok(pred, pred_label) if gt_label == "Passable" else None
    obstacle_ok = obstacle_semantic_match is True
    distance_ok = (
        distance_mentioned
        and distance_analysis["distance_mean_abs_error_m"] is not None
        and distance_analysis["distance_mean_abs_error_m"] <= 0.5
    ) if parse_success else False
    compliance_score = (
        binary_score(blocked_ok, default_when_none=1.0)
        * binary_score(highrisk_ok, default_when_none=1.0)
        * binary_score(passable_ok, default_when_none=1.0)
    )
    ias_components = {
        "label": (0.25, binary_score(label_match)),
        "obstacle": (0.15, mention_score(obstacle_mentioned, obstacle_semantic_match)),
        "direction": (0.10, binary_score(side_match, default_when_none=1.0)),
        "distance": (0.10, bounded_error_score(distance_analysis["distance_mean_abs_error_m"], tolerance=0.5, default_when_missing=0.0)),
        "safety": (0.40, compliance_score),
    }
    ias_score = weighted_average(ias_components)
    strict_match = (
        parse_success
        and label_match
        and obstacle_ok
        and distance_ok
        and distance_mentioned
        and (side_match is True if side_evaluated else True)
        and (blocked_ok is True if gt_label == "Blocked" else True)
        and (highrisk_ok is True if gt_label == "HighRisk" else True)
        and (passable_ok is True if gt_label == "Passable" else True)
    )
    success = strict_match
    return {
        "scene_name": scene_name,
        "traj_name": traj_name,
        "ground_truth_label": gt_label,
        "predicted_label": pred_label,
        "parse_success": parse_success,
        "label_match": label_match,
        "side_evaluated": side_evaluated,
        "side_match": side_match,
        "obstacle_mentioned": obstacle_mentioned,
        "obstacle_semantic_evaluated": obstacle_semantic_evaluated,
        "obstacle_semantic_match": obstacle_semantic_match,
        "distance_mentioned": distance_mentioned,
        "distance_ok": distance_ok,
        **distance_analysis,
        "ias_score": ias_score,
        "blocked_compliance": blocked_ok,
        "highrisk_compliance": highrisk_ok,
        "passable_accessibility_ok": passable_ok,
        "strict_match": strict_match,
        "success": success,
        "ground_truth_instruction": gt,
        "predicted_instruction": pred,
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(results)
    label_match = sum(1 for r in results if r["label_match"])
    obstacle_mentioned = sum(1 for r in results if r["obstacle_mentioned"])
    obstacle_semantic_eval = [r for r in results if r["obstacle_semantic_evaluated"]]
    obstacle_semantic_match = sum(1 for r in obstacle_semantic_eval if r["obstacle_semantic_match"])
    distance_mentioned = sum(1 for r in results if r["distance_mentioned"])
    distance_eval = [r for r in results if r["distance_pair_count"] > 0]
    side_eval = [r for r in results if r["side_evaluated"]]
    side_match = sum(1 for r in side_eval if r["side_match"])
    blocked_cases = [r for r in results if r["ground_truth_label"] == "Blocked"]
    blocked_ok = sum(1 for r in blocked_cases if r["blocked_compliance"])
    highrisk_cases = [r for r in results if r["ground_truth_label"] == "HighRisk"]
    highrisk_ok = sum(1 for r in highrisk_cases if r["highrisk_compliance"])
    passable_cases = [r for r in results if r["ground_truth_label"] == "Passable"]
    passable_ok = sum(1 for r in passable_cases if r["passable_accessibility_ok"])
    strict_match = sum(1 for r in results if r["strict_match"])
    success = sum(1 for r in results if r["success"])
    ias_mean = sum(r["ias_score"] for r in results) / count if count else 0.0
    return {
        "trajectory_count": count,
        "accessibility_accuracy": label_match / count if count else 0.0,
        "obstacle_mention_rate": obstacle_mentioned / count if count else 0.0,
        "obstacle_semantic_eval_count": len(obstacle_semantic_eval),
        "obstacle_semantic_match_rate": obstacle_semantic_match / len(obstacle_semantic_eval) if obstacle_semantic_eval else None,
        "pass_side_eval_count": len(side_eval),
        "pass_side_accuracy": side_match / len(side_eval) if side_eval else None,
        "blocked_case_count": len(blocked_cases),
        "blocked_compliance_rate": blocked_ok / len(blocked_cases) if blocked_cases else None,
        "highrisk_case_count": len(highrisk_cases),
        "highrisk_compliance_rate": highrisk_ok / len(highrisk_cases) if highrisk_cases else None,
        "passable_case_count": len(passable_cases),
        "passable_accessibility_rate": passable_ok / len(passable_cases) if passable_cases else None,
        "distance_mention_rate": distance_mentioned / count if count else 0.0,
        "distance_eval_count": len(distance_eval),
        "distance_mean_abs_error_m": (sum(r["distance_mean_abs_error_m"] for r in distance_eval if r["distance_mean_abs_error_m"] is not None) / len(distance_eval)) if distance_eval else None,
        "SNSR": strict_match / count if count else 0.0,
        "success_rate": success / count if count else 0.0,
        "IAS": ias_mean,
        "ground_truth_label_counts": dict(Counter(r["ground_truth_label"] for r in results)),
        "predicted_label_counts": dict(Counter(r["predicted_label"] for r in results)),
    }


def evaluate_dataset(artifacts_root: Path, model_key: str, prompt_id: str, scene_names: list[str]) -> dict[str, Any]:
    results = []
    scene_summaries: dict[str, Any] = {}
    eval_root = artifacts_root / "evaluations" / "wheelchair" / model_key / prompt_id
    for scene_name in scene_names:
        scene_results = []
        for ctx_path in sorted((artifacts_root / "contexts" / scene_name).glob("*.json")):
            traj_name = ctx_path.stem
            pred_path = artifacts_root / "predictions" / "wheelchair" / model_key / prompt_id / scene_name / f"{traj_name}.json"
            if not pred_path.exists():
                continue
            r = evaluate_traj(artifacts_root, scene_name, traj_name, model_key, prompt_id)
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
