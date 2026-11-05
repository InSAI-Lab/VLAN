from __future__ import annotations

import argparse
from pathlib import Path
import sys

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vlan_final.eval.action import evaluate_dataset as evaluate_action_dataset
from vlan_final.eval.blind import evaluate_dataset as evaluate_blind_dataset
from vlan_final.eval.pedestrian import evaluate_dataset as evaluate_pedestrian_dataset
from vlan_final.eval.wheelchair import evaluate_dataset as evaluate_wheelchair_dataset
from vlan_final.tasks.defaults import DEFAULT_ARTIFACTS_ROOT, DEFAULT_DATA_ROOT, default_prompt_id


def _load_yaml(path_str: str | None) -> dict:
    if not path_str:
        return {}
    path = Path(path_str)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--task", type=str, default="")
    parser.add_argument("--model-config", type=str, default="")
    parser.add_argument("--prompt-id", type=str, default="")
    parser.add_argument("--data-root", type=str, default="")
    parser.add_argument("--artifacts-root", type=str, default="")
    parser.add_argument("--scene_names", type=str, default="")
    args = parser.parse_args()

    cfg = _load_yaml(args.config)
    task = args.task or cfg.get("task", "")
    if not task:
        raise ValueError("Task is required. Provide --task or set task in --config.")
    prompt_id = args.prompt_id or cfg.get("prompt_id") or default_prompt_id(task)
    model_config_path = args.model_config or cfg.get("model_config", "")
    if not model_config_path:
        raise ValueError("Model config is required. Provide --model-config or set model_config in --config.")
    model_cfg = yaml.safe_load(_resolve_path(model_config_path).read_text(encoding="utf-8"))
    model_key = model_cfg["model_key"]
    data_root = _resolve_path(args.data_root or cfg.get("data_root", DEFAULT_DATA_ROOT))
    artifacts_root = _resolve_path(args.artifacts_root or cfg.get("artifacts_root", DEFAULT_ARTIFACTS_ROOT))
    scene_names = [s.strip() for s in args.scene_names.split(",") if s.strip()] if args.scene_names else cfg.get("scene_names")
    if not scene_names:
        scene_names = sorted(p.name for p in data_root.iterdir() if p.is_dir())

    if task in {"pedestrian", "humanoid"}:
        summary = evaluate_pedestrian_dataset(data_root, artifacts_root, model_key, prompt_id, scene_names, task_name=task)
        overall = summary["overall"]
        print(
            f"[HumanoidEval] model={model_key} prompt={prompt_id} "
            f"scenes={summary['scene_count']} traj={summary['trajectory_count']} "
            f"strict={overall['strict_match_rate']:.4f} label={overall['label_match_rate']:.4f} "
            f"side={overall['side_match_rate'] if overall['side_match_rate'] is not None else 'NA'}"
        )
        return

    if task == "blind":
        summary = evaluate_blind_dataset(artifacts_root, model_key, prompt_id, scene_names)
        overall = summary["overall"]
        print(
            f"[BlindEval] model={model_key} prompt={prompt_id} "
            f"scenes={summary['scene_count']} traj={summary['trajectory_count']} "
            f"snsr={overall['SNSR']:.4f} label={overall['accessibility_accuracy']:.4f} "
            f"clock={overall['clock_direction_accuracy'] if overall['clock_direction_accuracy'] is not None else 'NA'}"
        )
        return

    if task == "wheelchair":
        summary = evaluate_wheelchair_dataset(artifacts_root, model_key, prompt_id, scene_names)
        overall = summary["overall"]
        print(
            f"[WheelchairEval] model={model_key} prompt={prompt_id} "
            f"scenes={summary['scene_count']} traj={summary['trajectory_count']} "
            f"snsr={overall['SNSR']:.4f} label={overall['accessibility_accuracy']:.4f} "
            f"side={overall['pass_side_accuracy'] if overall['pass_side_accuracy'] is not None else 'NA'}"
        )
        return

    if task in {"drone", "quadrupedRobot"}:
        summary = evaluate_action_dataset(artifacts_root, data_root, task, model_key, prompt_id, scene_names)
        overall = summary["overall"]["overall"]
        print(
            f"[ActionEval] task={task} model={model_key} prompt={prompt_id} "
            f"scenes={summary['scene_count']} traj={summary['trajectory_count']} "
            f"success={overall['success_rate']:.4f} goal={overall['goal_reached_rate']:.4f} "
            f"collision_free={overall['collision_free_rate']:.4f} "
            f"spl={overall['spl']:.4f} safe_spl={overall['safe_spl']:.4f}"
        )
        return

    raise ValueError(f"Unsupported task: {task}")


if __name__ == "__main__":
    main()
