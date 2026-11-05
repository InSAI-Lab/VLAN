from __future__ import annotations

DEFAULT_PROMPT_ID_BY_TASK = {
    "pedestrian": "pedestrian_v3_6_frozen",
    "humanoid": "pedestrian_v3_6_frozen",
    "drone": "drone_action_main",
    "quadrupedRobot": "quadruped_action_main",
    "blind": "blind_instruction_main",
    "wheelchair": "wheelchair_instruction_main",
}

DEFAULT_DATA_ROOT = "data/test"
DEFAULT_ARTIFACTS_ROOT = "artifacts"
DEFAULT_MAX_NEW_TOKENS = 256


def default_prompt_id(task: str) -> str:
    if task not in DEFAULT_PROMPT_ID_BY_TASK:
        raise ValueError(f"Unsupported task: {task}")
    return DEFAULT_PROMPT_ID_BY_TASK[task]
