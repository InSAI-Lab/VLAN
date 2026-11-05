from __future__ import annotations

from typing import Any


def render_wheelchair_prompt(context: dict[str, Any]) -> str:
    user = context["users"]["wheelchair"]
    goal_summary = user["goal_summary"].replace("very_near", "very near")
    obstacle_summary = user["obstacle_summary"].replace("very_near", "very near")
    return "\n".join(
        [
            "You are generating a navigation instruction for a wheelchair user from three first-person keyframes.",
            "You must infer the accessibility label from the images and the motion context.",
            "Choose the predicted_accessibility_label from exactly these options: Passable, Bypass, HighRisk, Blocked.",
            "Use these definitions when predicting accessibility:",
            "Passable: the route is wide enough, smooth enough, and clear enough that the wheelchair can continue mostly straight with only a small adjustment.",
            "Bypass: the wheelchair can still pass, but it needs a clear detour around the obstacle with enough width and stable surface.",
            "HighRisk: the route may still be usable, but the clearance, width, or surface quality makes wheelchair travel difficult or uncertain. The instruction should explicitly mention caution.",
            "Blocked: the route is not safely passable for a wheelchair. The instruction must tell the user to stop and choose another accessible route.",
            "Apply these decision rules strictly:",
            "Use Passable only when the route still looks comfortably usable for a wheelchair without a real detour.",
            "Use Bypass only when there is a believable side detour with enough wheelchair clearance.",
            "Use HighRisk when the route may be narrow, unstable, uneven, or difficult, even if it is not fully blocked.",
            "Use Blocked when the route is not safely passable, including signs or hazards that prevent forward travel.",
            "Do not default to Passable just because the obstacle is visible but small.",
            "Minimal examples:",
            "Passable example: Roll forward about 2.0 meters toward the goal. The route remains clear enough for the wheelchair.",
            "Bypass example: Roll forward about 1.0 meters. As you approach the trash bin, keep slightly to the left to go around it while maintaining clear wheelchair space. Then continue about 1.2 more meters.",
            "HighRisk example: Move forward cautiously for about 1.0 meters. The route ahead is narrow or unstable for a wheelchair, so slow down and continue only if the clearance remains safe.",
            "Blocked example: Stop. The route ahead is not safely passable for a wheelchair. Choose another accessible route.",
            "You must output three fields: predicted_accessibility_label, predicted_obstacle_description, and navigation_instruction.",
            "The field predicted_obstacle_description must be a concrete English noun phrase for the main obstacle.",
            "Do not use generic phrases such as 'obstacle', 'object', 'something ahead', or 'hazard' by themselves.",
            "The navigation_instruction must contain explicit metric distances in meters.",
            "For wheelchair users, be conservative about clearance, surface quality, and route width.",
            "Passable means the route is reasonably clear and can be followed mostly straight with only a small adjustment.",
            "Bypass means the route is usable but requires a clear detour around the obstacle.",
            "HighRisk means the route may be narrow, uneven, unstable, or difficult for a wheelchair and should include cautionary wording.",
            "Blocked means the route is not safely passable for a wheelchair and the instruction must tell the user to stop and choose another accessible route.",
            "For Bypass, prefer a three-part structure:",
            "1. an initial forward distance in meters,",
            "2. a side-specific avoidance action that mentions both the obstacle and the passing side,",
            "3. a continuation phrase after passing the obstacle, preferably with another distance or a continue-toward-the-goal phrase.",
            "If a safer passing side is provided in context, use that side explicitly and do not flip left and right.",
            "For Passable, do not overreact or describe a full detour unless clearly required.",
            "Do not default to Bypass when the route appears narrow, uneven, blocked, or unsafe for a wheelchair. Use HighRisk or Blocked when appropriate.",
            "If you predict Blocked, the navigation_instruction must begin with Stop.",
            "If you predict HighRisk, the navigation_instruction must explicitly use caution wording such as cautious, slow down, narrow, unstable, uneven, or clearance.",
            "Output JSON only in the format: "
            "{\"predicted_accessibility_label\": \"...\", \"predicted_obstacle_description\": \"...\", \"navigation_instruction\": \"...\"}",
            "",
            user["prefix_motion_summary"],
            goal_summary,
            obstacle_summary,
        ]
    )
