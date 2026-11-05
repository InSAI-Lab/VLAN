from __future__ import annotations

from typing import Any


def render_pedestrian_v3_6(context: dict[str, Any]) -> str:
    user = context["users"].get("humanoid") or context["users"]["pedestrian"]
    label = user["accessibility_label"]
    goal_summary = user["goal_summary"].replace("very_near", "very near")
    obstacle_summary = user["obstacle_summary"].replace("very_near", "very near")
    return "\n".join(
        [
            "You are generating a humanoid navigation output from three first-person keyframes.",
            "You must output both a concrete obstacle description and a navigation instruction.",
            "Write one concise but reasonably detailed English navigation instruction.",
            "The instruction must contain explicit metric distances in meters, not vague phrases.",
            "Use the provided goal and obstacle distance estimates as reliable references for scale.",
            "You are not told the obstacle category or object name. Infer it from the images.",
            "The field predicted_obstacle_description must be a concrete English noun phrase for the main obstacle, such as 'traffic cone', 'construction fence', 'trash bin', 'bench', 'bicycle', 'scooter', 'motorcycle', 'speed bump', 'low overhead box obstacle', 'tree branch', 'traffic sign', or 'drop-off hazard area'.",
            "Do not use generic phrases such as 'obstacle', 'object', 'something ahead', 'hazard', or 'barrier' by themselves.",
            "If the accessibility label is Blocked, the instruction must tell the humanoid user to stop and must not describe any turning, bypass, detour, or continued forward motion.",
            "For Blocked, valid outputs are short stop instructions only, such as stopping and choosing another route. Do not add a turn direction.",
            "If the accessibility label is Passable, keep the motion mostly straight and include one approximate forward distance in meters. A small adjustment is acceptable, but do not describe a full detour unless the images clearly require it.",
            "If a safer passing side is specified, your navigation instruction must use that same side explicitly. Do not flip left and right.",
            "For Bypass, do not output only a short phrase like 'turn left', 'turn right', 'pass on the left side', or 'go around the obstacle'.",
            "A correct Bypass instruction should usually contain three parts in this order:",
            "1. a forward distance before reaching the obstacle,",
            "2. a side-specific avoidance action that mentions both the obstacle description and the passing side,",
            "3. a continuation phrase after passing the obstacle, preferably with another distance or an explicit continue-toward-the-goal phrase.",
            "For Bypass, include two distances whenever possible: one distance before the avoidance maneuver and one distance after passing the obstacle.",
            "The first distance before changing side should stay close to the obstacle distance estimate, usually within about 0.3 meters of that obstacle distance.",
            "When describing the bypass, mention both the obstacle description and the passing side in the same clause.",
            "Preferred style for Bypass:",
            "\"Walk forward about <d1> meters. As you approach the <obstacle>, keep slightly to the <left/right> to go around it. After passing it, continue toward the goal for about <d2> more meters.\"",
            "Use decimal distances such as 0.8 meters, 1.2 meters, or 2.1 meters when appropriate.",
            "Avoid vague distance phrases such as 'a bit', 'a short distance', 'nearby', or 'until you reach it' unless they are accompanied by a numeric distance in meters.",
            "Do not mention coordinates, dx/dz/yaw, or frame numbers.",
            "Output JSON only in the format: {\"predicted_obstacle_description\": \"...\", \"navigation_instruction\": \"...\"}",
            "",
            f"Accessibility label for the humanoid user: {label}.",
            user["prefix_motion_summary"],
            goal_summary,
            obstacle_summary,
        ]
    )
