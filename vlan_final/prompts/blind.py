from __future__ import annotations

from typing import Any


def render_blind_prompt(context: dict[str, Any]) -> str:
    user = context["users"]["blind"]
    return "\n".join(
        [
            "You are generating a navigation instruction for a blind user from three first-person keyframes.",
            "You must infer the accessibility label from the images and the motion context.",
            "Choose the predicted_accessibility_label from exactly these options: Passable, Bypass, HighRisk, Blocked.",
            "Use these definitions when predicting accessibility:",
            "Passable: the route is mostly clear and can be followed without a dedicated detour.",
            "Bypass: the route is still usable, but the user should go around a localized obstacle from the safer clock direction.",
            "HighRisk: the route may still be usable, but there is elevated risk for a blind user, such as a surface change, low obstacle, overhead obstacle, sign, narrow passage, or uncertain clearance. The instruction should explicitly warn the user to slow down and verify the path, and it should mention the specific risk source.",
            "Blocked: the route is not safe to continue. The instruction must tell the user to stop and seek another safe route.",
            "Apply these decision rules strictly:",
            "If the route directly ahead is not safely traversable, predict Blocked.",
            "If there is a low overhead obstacle, hanging branch, surface drop, sign blocking the route, or uncertain narrow clearance, do not default to Passable.",
            "Use HighRisk when the user may still continue only with caution and verification.",
            "Use Bypass only when the user can continue by going around a localized obstacle from a safer clock direction.",
            "Minimal examples:",
            "Blocked example: Stop. The route ahead at 12 o'clock is blocked. Do not continue forward.",
            "HighRisk example: Slow down. There is a risky low branch ahead at head height. Check the area carefully before continuing.",
            "Bypass example: Walk 1 step toward 12 o'clock. There is an obstacle ahead. Angle toward 10 o'clock to go around it. Then continue 2 more steps.",
            "You must output three fields: predicted_accessibility_label, predicted_obstacle_description, and navigation_instruction.",
            "The field predicted_obstacle_description must be a concrete English noun phrase for the main obstacle.",
            "Do not use generic phrases such as 'obstacle', 'object', 'something ahead', or 'hazard' by themselves.",
            "The navigation_instruction must use step counts instead of meters.",
            "The navigation_instruction must use clock directions instead of left/right words.",
            "Use expressions such as '12 o'clock', '10 o'clock', or '2 o'clock'.",
            "Do not use metric distances such as meters or m.",
            "Do not use left or right inside the navigation_instruction.",
            "Do not write the instruction as if the user is visually tracking the obstacle.",
            "Avoid phrases such as 'as you approach the trash bin' or 'when you reach the cone'.",
            "Prefer phrases such as 'there is a trash bin ahead' or 'there is a low branch ahead', followed by step counts and clock-direction guidance.",
            "For Bypass, prefer a three-part structure:",
            "1. an initial step count before reaching the obstacle,",
            "2. a clock-direction avoidance action that mentions both the obstacle description and the passing direction,",
            "3. a continuation phrase with another step count after passing the obstacle.",
            "For HighRisk, explicitly warn the user to slow down, scan carefully, or confirm the path before continuing.",
            "For HighRisk, mention the specific high-risk information when possible, such as low branch, overhead obstacle, surface change, sign blocking the route, or drop-off hazard.",
            "For Blocked, output a short stop instruction only. Do not describe a continued forward route.",
            "If you predict Blocked, the navigation_instruction must begin with Stop.",
            "For Passable, keep the route mostly straight and do not overstate the need for a detour.",
            "If a safer passing direction is provided in context, use that direction explicitly and do not contradict it.",
            "Do not default to Passable when the obstacle appears directly in the walking line or when the path looks uncertain.",
            "Output JSON only in the format: "
            "{\"predicted_accessibility_label\": \"...\", \"predicted_obstacle_description\": \"...\", \"navigation_instruction\": \"...\"}",
            "",
            user["prefix_motion_summary"],
            user["goal_summary"],
            user["obstacle_summary"],
        ]
    )
