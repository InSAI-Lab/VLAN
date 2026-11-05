from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any

from vlan_final.eval.ias import binary_score, bounded_error_score, mention_score, weighted_average
from vlan_final.eval.pedestrian import (
    CANONICAL_ALIASES,
    GENERIC_OBSTACLE_TERMS,
    blocked_hard_fail_from_instruction,
    canonical_obstacle_class,
    extract_predicted_obstacle_class,
    has_stop_intent,
    normalize_predicted_label,
    normalize_text,
    predicted_label_from_instruction,
)
from vlan_final.utils.io import load_json, save_json


def mentions_obstacle(text: str) -> bool:
    t = normalize_text(text)
    if extract_predicted_obstacle_class(t) is not None:
        return True
    return any(term in t for term in GENERIC_OBSTACLE_TERMS)


def extract_step_values(text: str) -> list[int]:
    return [int(m.group(1)) for m in re.finditer(r"\b(\d+)\s*steps?\b", normalize_text(text))]


def step_errors(gt_text: str, pred_text: str) -> dict[str, Any]:
    gt_vals = extract_step_values(gt_text)
    pred_vals = extract_step_values(pred_text)
    pair_count = min(len(gt_vals), len(pred_vals))
    abs_errors = [abs(gt_vals[i] - pred_vals[i]) for i in range(pair_count)]
    return {
        "gt_step_values": gt_vals,
        "pred_step_values": pred_vals,
        "step_pair_count": pair_count,
        "step_abs_errors": abs_errors,
        "step_mean_abs_error": (sum(abs_errors) / pair_count) if pair_count else None,
    }


def has_step_reference(text: str) -> bool:
    return bool(re.search(r"\b\d+\s*steps?\b", normalize_text(text)))


def leaks_metric_units(text: str) -> bool:
    return bool(re.search(r"\b\d+(\.\d+)?\s*(meter|meters|m)\b", normalize_text(text)))


def extract_clock_hours(text: str) -> set[int]:
    hours = set()
    for m in re.finditer(r"\b(\d{1,2})\s*o['’]?\s*clock\b", normalize_text(text)):
        h = int(m.group(1))
        if h == 0:
            h = 12
        if 1 <= h <= 12:
            hours.add(h)
    return hours


def expected_clock_hours(pass_side: str) -> set[int] | None:
    if pass_side == "left":
        return {10, 11}
    if pass_side == "right":
        return {1, 2}
    return None


def highrisk_compliant(text: str) -> bool:
    t = normalize_text(text)
    keywords = ["slow down", "cautious", "careful", "scan", "check", "confirm", "cane", "high-risk", "high risk"]
    return any(k in t for k in keywords)


def visual_trigger_free(text: str) -> bool:
    t = normalize_text(text)
    banned = [
        "when you reach",
        "as you approach",
        "once you reach",
        "when you get to",
    ]
    return not any(p in t for p in banned)


def risk_source_mentioned(text: str) -> bool:
    t = normalize_text(text)
    keywords = ["branch", "overhead", "surface", "sign", "drop-off", "drop off", "pit", "head height", "shoulder height"]
    return any(k in t for k in keywords)


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
    pred_file = artifacts_root / "predictions" / "blind" / model_key / prompt_id / scene_name / f"{traj_name}.json"
    pred_wrap = load_json(pred_file)
    parsed_output = _normalize_instruction_prediction(pred_wrap.get("parsed_output", {}))
    pred = parsed_output.get("navigation_instruction", "")
    gt = target["users"]["blind"]["ground_truth_instruction"]
    gt_label = target["users"]["blind"]["accessibility_label"]
    gt_side = target["users"]["blind"]["pass_side"]
    obstacle = target["obstacle"]

    parse_success = isinstance(pred, str) and len(pred.strip()) > 0
    structured_label = normalize_predicted_label(parsed_output.get("predicted_accessibility_label"))
    pred_label = structured_label if structured_label is not None else (predicted_label_from_instruction(pred) if parse_success else "ParseFailed")
    gt_obstacle_class = canonical_obstacle_class(obstacle)
    obstacle_text_for_eval = parsed_output.get("predicted_obstacle_description") or pred
    pred_obstacle_class = extract_predicted_obstacle_class(obstacle_text_for_eval) if parse_success else None
    label_match = pred_label == gt_label
    obstacle_mentioned = mentions_obstacle(obstacle_text_for_eval) if parse_success else False
    obstacle_semantic_evaluated = pred_obstacle_class is not None
    obstacle_semantic_match = (pred_obstacle_class == gt_obstacle_class) if obstacle_semantic_evaluated else None
    step_mentioned = has_step_reference(pred) if parse_success else False
    metric_leak = leaks_metric_units(pred) if parse_success else False
    step_analysis = step_errors(gt, pred) if parse_success else {
        "gt_step_values": [],
        "pred_step_values": [],
        "step_pair_count": 0,
        "step_abs_errors": [],
        "step_mean_abs_error": None,
    }
    expected_hours = expected_clock_hours(gt_side) if gt_label in {"Bypass", "HighRisk"} else None
    pred_hours = extract_clock_hours(pred) if parse_success else set()
    clock_evaluated = expected_hours is not None
    clock_match = (len(pred_hours.intersection(expected_hours)) > 0) if clock_evaluated else None
    highrisk_ok = highrisk_compliant(pred) if gt_label == "HighRisk" else None
    blocked_ok = has_stop_intent(pred) and not blocked_hard_fail_from_instruction(pred) if gt_label == "Blocked" else None
    obstacle_ok = obstacle_semantic_match is True
    step_ok = (
        step_mentioned
        and step_analysis["step_mean_abs_error"] is not None
        and step_analysis["step_mean_abs_error"] <= 0.5
    )
    risk_ok = risk_source_mentioned(pred) if gt_label == "HighRisk" else None
    safety_ok = (
        (highrisk_ok is True if gt_label == "HighRisk" else True)
        and (blocked_ok is True if gt_label == "Blocked" else True)
        and not metric_leak
    )
    ias_components = {
        "label": (0.20, binary_score(label_match)),
        "obstacle": (0.15, mention_score(obstacle_mentioned, obstacle_semantic_match)),
        "steps": (0.15, bounded_error_score(step_analysis["step_mean_abs_error"], tolerance=0.5, default_when_missing=0.0)),
        "clock": (0.20, binary_score(clock_match, default_when_none=1.0)),
        "safety": (0.20, 1.0 if safety_ok and (risk_ok is True if gt_label == "HighRisk" else True) else 0.0),
        "style": (0.10, 1.0 if visual_trigger_free(pred) else 0.0),
    }
    ias_score = weighted_average(ias_components)
    strict_match = (
        parse_success
        and label_match
        and obstacle_ok
        and step_ok
        and not metric_leak
        and (clock_match is True if clock_evaluated else True)
        and (highrisk_ok is True if gt_label == "HighRisk" else True)
        and (risk_ok is True if gt_label == "HighRisk" else True)
        and (blocked_ok is True if gt_label == "Blocked" else True)
        and visual_trigger_free(pred)
    )
    success = strict_match
    return {
        "scene_name": scene_name,
        "traj_name": traj_name,
        "ground_truth_label": gt_label,
        "predicted_label": pred_label,
        "parse_success": parse_success,
        "label_match": label_match,
        "ground_truth_pass_side": gt_side,
        "clock_evaluated": clock_evaluated,
        "clock_match": clock_match,
        "pred_clock_hours": sorted(pred_hours),
        "obstacle_mentioned": obstacle_mentioned,
        "obstacle_semantic_evaluated": obstacle_semantic_evaluated,
        "obstacle_semantic_match": obstacle_semantic_match,
        "step_mentioned": step_mentioned,
        "step_ok": step_ok,
        "metric_leak": metric_leak,
        **step_analysis,
        "ias_score": ias_score,
        "highrisk_compliance": highrisk_ok,
        "risk_source_mentioned": risk_ok,
        "blocked_compliance": blocked_ok,
        "strict_match": strict_match,
        "success": success,
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
    step_mentioned = sum(1 for r in results if r["step_mentioned"])
    step_eval = [r for r in results if r["step_pair_count"] > 0]
    clock_eval = [r for r in results if r["clock_evaluated"]]
    clock_match = sum(1 for r in clock_eval if r["clock_match"])
    highrisk_cases = [r for r in results if r["ground_truth_label"] == "HighRisk"]
    highrisk_ok = sum(1 for r in highrisk_cases if r["highrisk_compliance"])
    highrisk_risk_source = sum(1 for r in highrisk_cases if r["risk_source_mentioned"])
    blocked_cases = [r for r in results if r["ground_truth_label"] == "Blocked"]
    blocked_ok = sum(1 for r in blocked_cases if r["blocked_compliance"])
    strict_match = sum(1 for r in results if r["strict_match"])
    success = sum(1 for r in results if r["success"])
    metric_leak = sum(1 for r in results if r["metric_leak"])
    ias_mean = sum(r["ias_score"] for r in results) / count if count else 0.0
    return {
        "trajectory_count": count,
        "accessibility_accuracy": label_match / count if count else 0.0,
        "obstacle_mention_rate": obstacle_mentioned / count if count else 0.0,
        "obstacle_semantic_eval_count": len(obstacle_semantic_eval),
        "obstacle_semantic_match_rate": obstacle_semantic_match / len(obstacle_semantic_eval) if obstacle_semantic_eval else None,
        "step_mention_rate": step_mentioned / count if count else 0.0,
        "step_eval_count": len(step_eval),
        "step_mean_abs_error": (sum(r["step_mean_abs_error"] for r in step_eval if r["step_mean_abs_error"] is not None) / len(step_eval)) if step_eval else None,
        "clock_direction_eval_count": len(clock_eval),
        "clock_direction_accuracy": clock_match / len(clock_eval) if clock_eval else None,
        "highrisk_case_count": len(highrisk_cases),
        "highrisk_compliance_rate": highrisk_ok / len(highrisk_cases) if highrisk_cases else None,
        "risk_communication_rate": highrisk_risk_source / len(highrisk_cases) if highrisk_cases else None,
        "blocked_case_count": len(blocked_cases),
        "blocked_compliance_rate": blocked_ok / len(blocked_cases) if blocked_cases else None,
        "metric_leak_rate": metric_leak / count if count else 0.0,
        "SNSR": strict_match / count if count else 0.0,
        "success_rate": success / count if count else 0.0,
        "IAS": ias_mean,
        "ground_truth_label_counts": dict(Counter(r["ground_truth_label"] for r in results)),
        "predicted_label_counts": dict(Counter(r["predicted_label"] for r in results)),
    }


def evaluate_dataset(artifacts_root: Path, model_key: str, prompt_id: str, scene_names: list[str]) -> dict[str, Any]:
    results = []
    scene_summaries: dict[str, Any] = {}
    eval_root = artifacts_root / "evaluations" / "blind" / model_key / prompt_id
    for scene_name in scene_names:
        scene_results = []
        for ctx_path in sorted((artifacts_root / "contexts" / scene_name).glob("*.json")):
            traj_name = ctx_path.stem
            pred_path = artifacts_root / "predictions" / "blind" / model_key / prompt_id / scene_name / f"{traj_name}.json"
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
