from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
import yaml
from PIL import Image
from transformers import AutoImageProcessor, AutoProcessor, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vlan_final.models.backends import resolve_model_dir
from vlan_final.models.internvl_image import load_image_with_dynamic_patches
from vlan_final.prompts.renderer import render_prompt
from vlan_final.tasks.defaults import DEFAULT_ARTIFACTS_ROOT, DEFAULT_DATA_ROOT, default_prompt_id
from vlan_final.utils.io import load_json


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--scene-name", default="0001")
    parser.add_argument("--traj-name", default="01")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--artifacts-root", default=DEFAULT_ARTIFACTS_ROOT)
    parser.add_argument("--prompt-id", default="")
    args = parser.parse_args()

    model_cfg = yaml.safe_load(_resolve_path(args.model_config).read_text(encoding="utf-8"))
    model_dir = resolve_model_dir(model_cfg)
    backend = model_cfg["backend"]
    task = args.task
    prompt_id = args.prompt_id or default_prompt_id(task)
    data_root = _resolve_path(args.data_root)
    artifacts_root = _resolve_path(args.artifacts_root)
    scene_dir = data_root / args.scene_name
    context = load_json(artifacts_root / "contexts" / args.scene_name / f"{args.traj_name}.json")
    prompt = render_prompt(task, context)
    image_paths = [scene_dir / frame["rgb_path"] for frame in context["images"]]
    pil_images = [Image.open(path).convert("RGB") for path in image_paths]

    if backend == "qwen_vl":
        from qwen_vl_utils import process_vision_info

        processor = AutoProcessor.from_pretrained(str(model_dir), trust_remote_code=True)
        messages = [{"role": "user", "content": [{"type": "image", "image": str(p)} for p in image_paths] + [{"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        batch = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        print({"backend": backend, "task": task, "prompt_id": prompt_id, "keys": sorted(batch.keys())})
        return

    if backend == "internvl_chat":
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir),
            trust_remote_code=True,
            use_fast=False,
            fix_mistral_regex=True,
        )
        patches = load_image_with_dynamic_patches(
            image_paths[0],
            input_size=int(model_cfg.get("image_size", 448)),
            min_num=int(model_cfg.get("min_tiles_per_image", 1)),
            max_num=int(model_cfg.get("max_tiles_per_image", 6)),
            use_thumbnail=bool(model_cfg.get("use_thumbnail", True)),
        )
        question = "Image-1: <image>\n" + prompt
        print(
            {
                "backend": backend,
                "task": task,
                "prompt_id": prompt_id,
                "tokenizer": tokenizer.__class__.__name__,
                "patches_shape": tuple(patches.shape),
                "question_preview": question[:160],
            }
        )
        return

    if backend in {"idefics3_hf", "mllama_hf", "llava_onevision_hf"}:
        processor = AutoProcessor.from_pretrained(str(model_dir), trust_remote_code=True)
        messages = [{"role": "user", "content": [{"type": "image"} for _ in pil_images] + [{"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        batch = processor(text=[text], images=pil_images, padding=True, return_tensors="pt")
        print({"backend": backend, "task": task, "prompt_id": prompt_id, "keys": sorted(batch.keys())})
        return

    if backend == "molmo_hf":
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
        image_processor = AutoImageProcessor.from_pretrained(str(model_dir), trust_remote_code=True)
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        image_batch = image_processor(images=pil_images, return_tensors="pt")
        text_inputs = tokenizer([text], padding=True, return_tensors="pt")
        print(
            {
                "backend": backend,
                "task": task,
                "prompt_id": prompt_id,
                "text_keys": sorted(text_inputs.keys()),
                "image_keys": sorted(image_batch.keys()),
            }
        )
        return

    raise ValueError(f"Unsupported backend for smoke test: {backend}")


if __name__ == "__main__":
    main()
