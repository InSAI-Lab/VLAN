from __future__ import annotations

import argparse
from pathlib import Path
import sys

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vlan_final.models.registry import load_backend
from vlan_final.prompts.renderer import render_prompt
from vlan_final.tasks.defaults import DEFAULT_ARTIFACTS_ROOT, DEFAULT_DATA_ROOT, DEFAULT_MAX_NEW_TOKENS, default_prompt_id
from vlan_final.utils.dataset_io import traj_root
from vlan_final.utils.io import load_json, save_json


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


def _fallback_parsed_output(task: str, raw_text: str) -> dict:
    if task in {"pedestrian", "humanoid"}:
        return {
            "predicted_obstacle_description": "",
            "navigation_instruction": raw_text.strip(),
        }
    if task == "blind":
        return {
            "predicted_accessibility_label": "",
            "predicted_obstacle_description": "",
            "navigation_instruction": raw_text.strip(),
        }
    if task == "wheelchair":
        return {
            "predicted_accessibility_label": "",
            "predicted_obstacle_description": "",
            "navigation_instruction": raw_text.strip(),
        }
    if task in {"drone", "quadrupedRobot"}:
        return {"predicted_actions": []}
    raise ValueError(f"Unsupported task for fallback parsing: {task}")


def _normalize_parsed_output(task: str, raw_text: str, parsed: object) -> dict:
    if isinstance(parsed, dict):
        return parsed
    if isinstance(parsed, str):
        return _fallback_parsed_output(task, parsed.strip() or raw_text.strip())
    return _fallback_parsed_output(task, raw_text)


def _recover_raw_text_from_exception(exc: Exception) -> str:
    msg = str(exc)
    marker = "Failed to parse JSON from model output:\n"
    if marker in msg:
        return msg.split(marker, 1)[1].strip()
    return msg.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--task", type=str, default="")
    parser.add_argument("--model-config", type=str, default="")
    parser.add_argument("--prompt-id", type=str, default="")
    parser.add_argument("--data-root", type=str, default="")
    parser.add_argument("--artifacts-root", type=str, default="")
    parser.add_argument("--max-new-tokens", type=int, default=0)
    parser.add_argument("--scene_names", type=str, default="")
    parser.add_argument("--traj_names", type=str, default="")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    run_cfg = _load_yaml(args.config)
    task = args.task or run_cfg.get("task", "")
    if not task:
        raise ValueError("Task is required. Provide --task or set task in --config.")
    prompt_id = args.prompt_id or run_cfg.get("prompt_id") or default_prompt_id(task)
    model_config_path = args.model_config or run_cfg.get("model_config", "")
    if not model_config_path:
        raise ValueError("Model config is required. Provide --model-config or set model_config in --config.")
    model_cfg = yaml.safe_load(_resolve_path(model_config_path).read_text(encoding="utf-8"))
    backend = load_backend(model_cfg)
    data_root = _resolve_path(args.data_root or run_cfg.get("data_root", DEFAULT_DATA_ROOT))
    artifacts_root = _resolve_path(args.artifacts_root or run_cfg.get("artifacts_root", DEFAULT_ARTIFACTS_ROOT))
    max_new_tokens = int(args.max_new_tokens or run_cfg.get("max_new_tokens", model_cfg.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS)))

    scene_names = [s.strip() for s in args.scene_names.split(",") if s.strip()] if args.scene_names else run_cfg.get("scene_names")
    if not scene_names:
        scene_names = sorted(p.name for p in data_root.iterdir() if p.is_dir())
    traj_names_filter = {t.strip() for t in args.traj_names.split(",") if t.strip()}

    for scene_name in scene_names:
        scene_dir = data_root / scene_name
        context_dir = artifacts_root / "contexts" / scene_name
        for context_path in sorted(context_dir.glob("*.json")):
            traj_name = context_path.stem
            if traj_names_filter and traj_name not in traj_names_filter:
                continue
            context = load_json(context_path)
            prompt = render_prompt(task, context)
            render_path = artifacts_root / "prompts_rendered" / task / scene_name / f"{traj_name}.txt"
            render_path.parent.mkdir(parents=True, exist_ok=True)
            render_path.write_text(prompt, encoding="utf-8")
            out_path = artifacts_root / "predictions" / task / model_cfg["model_key"] / prompt_id / scene_name / f"{traj_name}.json"
            if out_path.exists() and not args.overwrite:
                print(f"[SKIP] {task} {model_cfg['model_key']} {scene_name}/{traj_name}")
                continue
            try:
                raw_text, parsed = backend.generate(traj_root(scene_dir, traj_name), context["images"], prompt, max_new_tokens=max_new_tokens)
            except Exception as exc:
                raw_text = _recover_raw_text_from_exception(exc)
                parsed = _fallback_parsed_output(task, raw_text)
            parsed = _normalize_parsed_output(task, raw_text, parsed)
            payload = {
                "scene_name": scene_name,
                "traj_name": traj_name,
                "task": task,
                "model_key": model_cfg["model_key"],
                "prompt_id": prompt_id,
                "raw_text": raw_text,
                "parsed_output": parsed,
            }
            save_json(out_path, payload)
            print(f"[OK] {task} {model_cfg['model_key']} {scene_name}/{traj_name}")


if __name__ == "__main__":
    main()
