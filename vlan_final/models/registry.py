from __future__ import annotations

from typing import Any

from .backends import (
    Idefics3Backend,
    InternVLBackend,
    LlavaOnevisionBackend,
    MllamaBackend,
    MolmoBackend,
    QwenVLBackend,
)


def load_backend(config: dict[str, Any]):
    backend = config.get("backend")
    if backend is None:
        model_key = str(config.get("model_key", ""))
        if model_key.startswith("qwen"):
            backend = "qwen_vl"
        elif model_key.startswith("internvl"):
            backend = "internvl_chat"
        else:
            raise ValueError(f"Could not infer backend for model_key={model_key}")
    if backend == "qwen_vl":
        return QwenVLBackend(config)
    if backend == "internvl_chat":
        return InternVLBackend(config)
    if backend == "idefics3_hf":
        return Idefics3Backend(config)
    if backend == "mllama_hf":
        return MllamaBackend(config)
    if backend == "llava_onevision_hf":
        return LlavaOnevisionBackend(config)
    if backend == "molmo_hf":
        return MolmoBackend(config)
    if backend == "gemini_proxy":
        raise NotImplementedError("Gemini proxy backend is not implemented.")
    raise ValueError(f"Unsupported backend: {backend}")
