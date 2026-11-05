from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import load_json


TASK_KEY_ALIASES = {
    "pedestrian": "humanoid",
    "humanoid": "humanoid",
    "blind": "blind",
    "bvi": "blind",
    "wheelchair": "wheelchair",
    "drone": "drone",
    "quadrupedRobot": "quadrupedRobot",
}

REVERSE_FALLBACK_KEYS = {
    "humanoid": ("humanoid", "pedestrian"),
    "blind": ("blind", "bvi"),
    "wheelchair": ("wheelchair",),
    "drone": ("drone",),
    "quadrupedRobot": ("quadrupedRobot",),
}


def canonical_task_key(task: str) -> str:
    try:
        return TASK_KEY_ALIASES[task]
    except KeyError as exc:
        raise ValueError(f"Unsupported task key: {task}") from exc


def scene_uses_new_layout(scene_dir: Path) -> bool:
    return any(p.is_dir() and p.name.isdigit() for p in scene_dir.iterdir())


def iter_traj_names(scene_dir: Path) -> list[str]:
    if scene_uses_new_layout(scene_dir):
        return sorted(p.name for p in scene_dir.iterdir() if p.is_dir() and p.name.isdigit())
    keyframes_root = scene_dir / "KeyFrames"
    if keyframes_root.exists():
        return sorted(p.name for p in keyframes_root.iterdir() if p.is_dir() and p.name.isdigit())
    return []


def traj_root(scene_dir: Path, traj_name: str) -> Path:
    new_root = scene_dir / traj_name
    return new_root if new_root.is_dir() else scene_dir


def metadata_path(scene_dir: Path, traj_name: str) -> Path:
    new_path = scene_dir / traj_name / "metadata.json"
    if new_path.exists():
        return new_path
    return scene_dir / "KeyFrames" / traj_name / "keyframes.json"


def trajectory_path(scene_dir: Path, traj_name: str) -> Path:
    new_path = scene_dir / traj_name / "trajectory.json"
    if new_path.exists():
        return new_path
    return scene_dir / "Trajectory" / f"{traj_name}.json"


def target_path(scene_dir: Path, traj_name: str) -> Path:
    return scene_dir / traj_name / "target.json"


def load_metadata(scene_dir: Path, traj_name: str) -> dict[str, Any]:
    path = metadata_path(scene_dir, traj_name)
    payload = load_json(path)
    if path.name == "metadata.json":
        return payload
    return {
        "scene_name": scene_dir.name,
        "traj_name": traj_name,
        "keyframes": payload,
    }


def _normalize_frame_rgb_path(frame: dict[str, Any], scene_dir: Path, traj_name: str) -> dict[str, Any]:
    out = dict(frame)
    rgb_path = str(out.get("rgb_path", ""))
    if (scene_dir / traj_name / "KeyFrames").exists() and rgb_path.startswith(f"KeyFrames/{traj_name}/rgb/"):
        out["rgb_path"] = rgb_path.replace(f"KeyFrames/{traj_name}/rgb/", "KeyFrames/", 1)
    elif (scene_dir / traj_name / "KeyFrames").exists() and rgb_path.startswith("rgb/"):
        out["rgb_path"] = f"KeyFrames/{Path(rgb_path).name}"
    return out


def load_keyframes_payload(scene_dir: Path, traj_name: str) -> dict[str, Any]:
    metadata = load_metadata(scene_dir, traj_name)
    payload = metadata["keyframes"]
    frames = [_normalize_frame_rgb_path(frame, scene_dir, traj_name) for frame in payload["frames"]]
    normalized = dict(payload)
    normalized["frames"] = frames
    return normalized


def load_frames(scene_dir: Path, traj_name: str) -> list[dict[str, Any]]:
    return load_keyframes_payload(scene_dir, traj_name)["frames"]


def load_trajectory(scene_dir: Path, traj_name: str) -> dict[str, Any]:
    return load_json(trajectory_path(scene_dir, traj_name))


def load_released_target(scene_dir: Path, traj_name: str) -> dict[str, Any] | None:
    path = target_path(scene_dir, traj_name)
    if not path.exists():
        return None
    return load_json(path)


def _normalize_accessibility_keys(payload: dict[str, Any]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for key, value in payload.items():
        task_key = canonical_task_key(key)
        if isinstance(value, dict):
            normalized[task_key] = str(value["label"])
        else:
            normalized[task_key] = str(value)
    return normalized


def load_accessibility_labels(scene_dir: Path, traj_name: str) -> dict[str, str]:
    trajectory = load_trajectory(scene_dir, traj_name)
    obstacle = trajectory["obstacles"][0]
    if "userAccessibility" in obstacle:
        return _normalize_accessibility_keys(obstacle["userAccessibility"])
    legacy = load_json(scene_dir / "KeyFrames" / traj_name / "accessibility_manual_v2_4label.json")
    return _normalize_accessibility_keys(legacy["per_user_accessibility"])


def target_user_payload(targets: dict[str, Any], task: str) -> dict[str, Any]:
    key = canonical_task_key(task)
    users = targets["users"]
    for candidate in REVERSE_FALLBACK_KEYS.get(key, (key, task)):
        if candidate in users:
            return users[candidate]
    raise KeyError(key)


def corrected_accessibility_label(targets: dict[str, Any], task: str) -> str:
    key = canonical_task_key(task)
    corrected = targets["corrected_accessibility"]
    for candidate in REVERSE_FALLBACK_KEYS.get(key, (key, task)):
        if candidate in corrected:
            return str(corrected[candidate])
    raise KeyError(key)
