from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    Idefics3ForConditionalGeneration,
    LlavaOnevisionForConditionalGeneration,
    MllamaForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
)


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    if "```" in text:
        for part in text.split("```"):
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            try:
                return json.loads(part)
            except Exception:
                continue
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        return json.loads(m.group(0))
    raise ValueError(f"Failed to parse JSON from model output:\n{text}")


REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_candidate(path_value: str | None) -> Path | None:
    if not path_value:
        return None
    p = Path(path_value)
    candidates = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append(REPO_ROOT / p)
    for cand in candidates:
        if (cand / "config.json").exists():
            return cand
    return candidates[0] if candidates else None


def resolve_model_dir(config: dict[str, Any]) -> Path:
    direct = _resolve_candidate(config.get("model_path") or config.get("local_model_dir"))
    if direct and (direct / "config.json").exists():
        return direct
    ckpt_dir = _resolve_candidate(config.get("ckpt_dir"))
    if ckpt_dir and (ckpt_dir / "config.json").exists():
        return ckpt_dir
    raise ValueError(f"Could not resolve model directory from config: {config.get('model_key')}")


def _resolve_dtype(config: dict[str, Any]) -> torch.dtype:
    dtype_name = str(config.get("torch_dtype", config.get("dtype", "bfloat16"))).lower()
    return torch.bfloat16 if dtype_name in {"bfloat16", "bf16"} else torch.float16


def _configure_generation(model: Any) -> None:
    if getattr(model, "generation_config", None) is None:
        return
    model.generation_config.do_sample = False
    for key in ("temperature", "top_p", "top_k"):
        if hasattr(model.generation_config, key):
            setattr(model.generation_config, key, None)


def _is_flash_attention_import_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "flashattention" in msg or "flash attention" in msg


def _load_model_with_fallback(model_cls: Any, model_dir: Path, kwargs: dict[str, Any]) -> Any:
    try:
        return model_cls.from_pretrained(str(model_dir), **kwargs)
    except ImportError as exc:
        if not _is_flash_attention_import_error(exc):
            raise
        fallback_kwargs = dict(kwargs)
        if "attn_implementation" in fallback_kwargs:
            fallback_kwargs["attn_implementation"] = "sdpa"
        if "use_flash_attn" in fallback_kwargs:
            fallback_kwargs["use_flash_attn"] = False
        return model_cls.from_pretrained(str(model_dir), **fallback_kwargs)


def _build_messages_and_images(scene_dir: Path, images: list[dict[str, Any]], prompt: str) -> tuple[list[dict[str, Any]], list[Image.Image]]:
    pil_images: list[Image.Image] = []
    content: list[dict[str, Any]] = []
    for frame in images:
        image_path = scene_dir / frame["rgb_path"]
        pil_image = Image.open(image_path).convert("RGB")
        pil_images.append(pil_image)
        content.append({"type": "image"})
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}], pil_images


class HFProcessorVisionBackend:
    def __init__(self, config: dict[str, Any], model: Any, processor: Any) -> None:
        self.config = config
        self.model = model
        self.processor = processor
        if hasattr(self.processor, "tokenizer") and self.processor.tokenizer is not None:
            self.processor.tokenizer.padding_side = "left"

    def generate(self, scene_dir: Path, images: list[dict[str, Any]], prompt: str, max_new_tokens: int) -> tuple[str, dict[str, Any]]:
        messages, pil_images = _build_messages_and_images(scene_dir, images, prompt)
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=pil_images, padding=True, return_tensors="pt")
        model_device = next(self.model.parameters()).device
        inputs = {k: v.to(model_device) if hasattr(v, "to") else v for k, v in inputs.items()}
        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        prompt_len = inputs["input_ids"].shape[1]
        raw_text = self.processor.batch_decode(
            generated_ids[:, prompt_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        return raw_text, extract_json(raw_text)


class QwenVLBackend:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        model_dir = resolve_model_dir(config)
        dtype = _resolve_dtype(config)
        model_cls = Qwen3VLForConditionalGeneration if config["model_key"].startswith("qwen3") else Qwen2_5_VLForConditionalGeneration
        self.model = _load_model_with_fallback(
            model_cls,
            model_dir,
            {
                "torch_dtype": dtype,
                "device_map": config.get("device_map", "auto"),
                "trust_remote_code": bool(config.get("trust_remote_code", True)),
                "attn_implementation": config.get("attn_implementation", "flash_attention_2"),
            },
        )
        _configure_generation(self.model)
        self.processor = AutoProcessor.from_pretrained(str(model_dir), trust_remote_code=True)
        if hasattr(self.processor, "tokenizer") and self.processor.tokenizer is not None:
            self.processor.tokenizer.padding_side = "left"
        self.model_dir = model_dir

    def generate(self, scene_dir: Path, images: list[dict[str, Any]], prompt: str, max_new_tokens: int) -> tuple[str, dict[str, Any]]:
        from qwen_vl_utils import process_vision_info

        content = []
        for frame in images:
            content.append({"type": "image", "image": str(scene_dir / frame["rgb_path"])})
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        model_device = next(self.model.parameters()).device
        inputs = {k: v.to(model_device) if hasattr(v, "to") else v for k, v in inputs.items()}
        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        prompt_len = inputs["input_ids"].shape[1]
        raw_text = self.processor.batch_decode(
            generated_ids[:, prompt_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        return raw_text, extract_json(raw_text)


class InternVLBackend:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        model_dir = resolve_model_dir(config)
        dtype = _resolve_dtype(config)
        self.model = _load_model_with_fallback(
            AutoModel,
            model_dir,
            {
                "torch_dtype": dtype,
                "low_cpu_mem_usage": True,
                "trust_remote_code": True,
                "device_map": config.get("device_map", "auto"),
                "use_flash_attn": bool(config.get("use_flash_attn", True)),
            },
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir),
            trust_remote_code=True,
            use_fast=False,
            fix_mistral_regex=True,
        )
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model_dir = model_dir

    def generate(self, scene_dir: Path, images: list[dict[str, Any]], prompt: str, max_new_tokens: int) -> tuple[str, dict[str, Any]]:
        from vlan_final.models.internvl_image import load_image_with_dynamic_patches

        input_size = int(self.config.get("image_size", 448))
        min_tiles_per_image = int(self.config.get("min_tiles_per_image", 1))
        max_tiles_per_image = int(self.config.get("max_tiles_per_image", 6))
        use_thumbnail = bool(self.config.get("use_thumbnail", True))
        pixel_value_list = []
        num_patches_list = []
        for frame in images:
            rgb_path = scene_dir / frame["rgb_path"]
            pixel_values = load_image_with_dynamic_patches(
                rgb_path,
                input_size=input_size,
                min_num=min_tiles_per_image,
                max_num=max_tiles_per_image,
                use_thumbnail=use_thumbnail,
            )
            num_patches_list.append(pixel_values.shape[0])
            pixel_value_list.append(pixel_values)
        model_param = next(self.model.parameters())
        all_pixel_values = torch.cat(pixel_value_list, dim=0).to(device=model_param.device, dtype=model_param.dtype)
        prefix = "\n".join([f"Image-{idx + 1}: <image>" for idx in range(len(images))])
        question = f"{prefix}\n{prompt}"
        generation_config = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "pad_token_id": self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id,
        }
        raw_text = self.model.chat(
            self.tokenizer,
            all_pixel_values,
            question,
            generation_config,
            num_patches_list=num_patches_list,
            history=None,
            return_history=False,
        )
        return raw_text, extract_json(raw_text)


class Idefics3Backend(HFProcessorVisionBackend):
    def __init__(self, config: dict[str, Any]) -> None:
        model_dir = resolve_model_dir(config)
        model = _load_model_with_fallback(
            Idefics3ForConditionalGeneration,
            model_dir,
            {
                "torch_dtype": _resolve_dtype(config),
                "device_map": config.get("device_map", "auto"),
                "trust_remote_code": bool(config.get("trust_remote_code", True)),
            },
        )
        _configure_generation(model)
        processor = AutoProcessor.from_pretrained(str(model_dir), trust_remote_code=True)
        self.model_dir = model_dir
        super().__init__(config, model, processor)


class MllamaBackend(HFProcessorVisionBackend):
    def __init__(self, config: dict[str, Any]) -> None:
        model_dir = resolve_model_dir(config)
        model = _load_model_with_fallback(
            MllamaForConditionalGeneration,
            model_dir,
            {
                "torch_dtype": _resolve_dtype(config),
                "device_map": config.get("device_map", "auto"),
                "trust_remote_code": bool(config.get("trust_remote_code", True)),
            },
        )
        _configure_generation(model)
        processor = AutoProcessor.from_pretrained(str(model_dir), trust_remote_code=True)
        self.model_dir = model_dir
        super().__init__(config, model, processor)


class LlavaOnevisionBackend(HFProcessorVisionBackend):
    def __init__(self, config: dict[str, Any]) -> None:
        model_dir = resolve_model_dir(config)
        model = _load_model_with_fallback(
            LlavaOnevisionForConditionalGeneration,
            model_dir,
            {
                "torch_dtype": _resolve_dtype(config),
                "device_map": config.get("device_map", "auto"),
                "trust_remote_code": bool(config.get("trust_remote_code", True)),
            },
        )
        _configure_generation(model)
        processor = AutoProcessor.from_pretrained(str(model_dir), trust_remote_code=True)
        self.model_dir = model_dir
        super().__init__(config, model, processor)


class MolmoBackend(HFProcessorVisionBackend):
    def __init__(self, config: dict[str, Any]) -> None:
        model_dir = resolve_model_dir(config)
        model = _load_model_with_fallback(
            AutoModelForImageTextToText,
            model_dir,
            {
                "torch_dtype": _resolve_dtype(config),
                "device_map": config.get("device_map", "auto"),
                "trust_remote_code": True,
            },
        )
        _configure_generation(model)
        self.config = config
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
        self.image_processor = AutoImageProcessor.from_pretrained(str(model_dir), trust_remote_code=True)
        self.model_dir = model_dir
        self.image_placeholder_token = "<|image|>"
        self.image_patch_token = "<im_patch>"
        self.im_start_token = "<im_start>"
        self.low_res_im_start_token = "<low_res_im_start>"
        self.im_end_token = "<im_end>"
        self.im_col_token = "<im_col>"
        self.image_use_col_tokens = True
        self.use_single_crop_col_tokens = False
        self.use_single_crop_start_token = True
        self.image_token_ids = [
            self.tokenizer.convert_tokens_to_ids(token)
            for token in [
                self.image_patch_token,
                self.im_col_token,
                self.im_start_token,
                self.low_res_im_start_token,
                self.im_end_token,
                "<image_low_res>",
            ]
            if self.tokenizer.convert_tokens_to_ids(token) is not None
        ]

    def _get_image_tokens(self, image_grid: np.ndarray) -> str:
        resized_h, resized_w, height, width = [int(x) for x in image_grid]
        per_row = [self.image_patch_token] * width
        if self.image_use_col_tokens:
            per_row = per_row + [self.im_col_token]
        joint: list[str] = [self.im_start_token] + per_row * height + [self.im_end_token]
        per_row_small = [self.image_patch_token] * resized_w
        use_single_crop_col_tokens = self.image_use_col_tokens if self.use_single_crop_col_tokens is None else self.use_single_crop_col_tokens
        image_start_token = self.low_res_im_start_token if self.use_single_crop_start_token else self.im_start_token
        if use_single_crop_col_tokens:
            per_row_small = per_row_small + [self.im_col_token]
        return "".join([image_start_token] + per_row_small * resized_h + [self.im_end_token] + joint)

    @staticmethod
    def _insert_bos(input_ids: np.ndarray, attention_mask: np.ndarray, bos_token_id: int, pad_token_id: int) -> tuple[np.ndarray, np.ndarray]:
        need_to_expand = len(input_ids.shape) == 1
        if need_to_expand:
            input_ids = input_ids[None, :]
            attention_mask = attention_mask[None, :]
        batch_size, seq_len = input_ids.shape
        if seq_len == 0:
            new_input_ids = np.full((batch_size, 1), bos_token_id, dtype=input_ids.dtype)
            new_attention_mask = np.ones((batch_size, 1), dtype=attention_mask.dtype)
            return (new_input_ids[0], new_attention_mask[0]) if need_to_expand else (new_input_ids, new_attention_mask)
        first_valid_index = (attention_mask == 1).argmax(axis=-1)
        bos_already_present = np.all(input_ids[np.arange(batch_size), first_valid_index] == bos_token_id)
        if bos_already_present:
            return (input_ids[0], attention_mask[0]) if need_to_expand else (input_ids, attention_mask)
        new_input_ids = np.full((batch_size, seq_len + 1), pad_token_id, dtype=input_ids.dtype)
        new_attention_mask = np.zeros((batch_size, seq_len + 1), dtype=attention_mask.dtype)
        src_idx = np.tile(np.arange(seq_len), (batch_size, 1))
        valid_mask = src_idx >= first_valid_index[:, None]
        tgt_idx = src_idx + 1
        batch_idx = np.tile(np.arange(batch_size)[:, None], (1, seq_len))
        new_input_ids[batch_idx[valid_mask], tgt_idx[valid_mask]] = input_ids[valid_mask]
        new_attention_mask[batch_idx[valid_mask], tgt_idx[valid_mask]] = 1
        new_input_ids[np.arange(batch_size), first_valid_index] = bos_token_id
        new_attention_mask[np.arange(batch_size), first_valid_index] = 1
        return (new_input_ids[0], new_attention_mask[0]) if need_to_expand else (new_input_ids, new_attention_mask)

    def generate(self, scene_dir: Path, images: list[dict[str, Any]], prompt: str, max_new_tokens: int) -> tuple[str, dict[str, Any]]:
        pil_images = [Image.open(scene_dir / frame["rgb_path"]).convert("RGB") for frame in images]
        messages = [{"role": "user", "content": [{"type": "image"} for _ in pil_images] + [{"type": "text", "text": prompt}]}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs = self.image_processor(images=pil_images, return_tensors="pt")
        image_grids = image_inputs["image_grids"]
        if hasattr(image_grids, "cpu"):
            image_grids = image_grids.cpu().numpy()
        text_with_tokens = text
        for image_grid in image_grids:
            text_with_tokens = text_with_tokens.replace(self.image_placeholder_token, self._get_image_tokens(image_grid), 1)
        text_inputs = self.tokenizer([text_with_tokens], padding=True, return_tensors="pt")
        input_ids = text_inputs["input_ids"].cpu().numpy()
        attention_mask = text_inputs["attention_mask"].cpu().numpy()
        bos = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
        input_ids, attention_mask = self._insert_bos(input_ids, attention_mask, bos, self.tokenizer.pad_token_id)
        token_type_ids = np.any(input_ids[:, :, None] == np.array(self.image_token_ids)[None, None, :], axis=-1)
        inputs = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.bool),
            **image_inputs,
        }
        model_device = next(self.model.parameters()).device
        inputs = {k: v.to(model_device) if hasattr(v, "to") else v for k, v in inputs.items()}
        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        prompt_len = inputs["input_ids"].shape[1]
        raw_text = self.tokenizer.batch_decode(
            generated_ids[:, prompt_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        return raw_text, extract_json(raw_text)
