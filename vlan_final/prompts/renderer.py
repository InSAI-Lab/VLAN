from __future__ import annotations

from typing import Any

from .blind import render_blind_prompt
from .drone import render_drone_prompt
from .pedestrian import render_pedestrian_v3_6
from .quadruped import render_quadruped_prompt
from .wheelchair import render_wheelchair_prompt


def render_prompt(task: str, context: dict[str, Any]) -> str:
    if task in {"pedestrian", "humanoid"}:
        return render_pedestrian_v3_6(context)
    if task == "blind":
        return render_blind_prompt(context)
    if task == "wheelchair":
        return render_wheelchair_prompt(context)
    if task == "drone":
        return render_drone_prompt(context)
    if task == "quadrupedRobot":
        return render_quadruped_prompt(context)
    raise ValueError(f"Unsupported task: {task}")
