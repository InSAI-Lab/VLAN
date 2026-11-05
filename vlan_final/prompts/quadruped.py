from __future__ import annotations

import math
from typing import Any


def _prefix_summary(observed_actions: list[dict[str, Any]]) -> str:
    if not observed_actions:
        return "No observed prefix actions are available."
    total_dx = sum(float(a["raw"]["dx"]) for a in observed_actions)
    total_dz = sum(float(a["raw"]["dz"]) for a in observed_actions)
    total_yaw = sum(float(a["raw"]["yaw"]) for a in observed_actions)
    prefix_distance = sum(math.hypot(float(a["raw"]["dx"]), float(a["raw"]["dz"])) for a in observed_actions)
    lateral = "stayed nearly centered"
    if total_dx > 0.08:
        lateral = "drifted slightly to the right"
    elif total_dx < -0.08:
        lateral = "drifted slightly to the left"
    heading = "kept a nearly stable heading"
    if total_yaw > 0.5:
        heading = "turned slightly to the right"
    elif total_yaw < -0.5:
        heading = "turned slightly to the left"
    return (
        f"In the first two observed actions, the quadruped robot already moved about {prefix_distance:.1f} meters forward in total, "
        f"{lateral}, and {heading}. The cumulative forward progress from those two actions is about {total_dz:.1f} meters."
    )


def render_quadruped_prompt(context: dict[str, Any]) -> str:
    quad = context["users"]["quadrupedRobot"]
    label = quad["accessibility_label"]
    label_guidance = {
        "Passable": "The obstacle is traversable for the quadruped robot. Stay goal-directed and avoid inserting an unnecessary detour.",
        "Bypass": "The obstacle should be bypassed on the required side. Establish clearance before obstacle depth, pass safely, then re-align toward the goal.",
        "HighRisk": "The route remains technically possible but risk is elevated. Keep progress smooth, conservative, and collision-free.",
        "Blocked": "The route should not continue forward. Predict a stop sequence for the next five actions, keeping dx, dz, and yaw close to zero.",
    }.get(label, "Stay goal-directed and keep the trajectory smooth.")
    stats = quad["observed_raw_action_stats"]
    rp = quad["rollout_start_pose_world"]["position_world"]
    yaw = quad["rollout_start_pose_world"]["yaw_world_deg"]
    return f"""
You are solving a visual navigation action prediction task for a quadruped robot.

You are given:
1. The first 3 RGB keyframes from a trajectory.
2. The first 2 executed RAW actions.
3. The rollout start pose after those observed actions.
4. The goal position.
5. The obstacle position and the required pass side if a bypass is needed.

Your job:
Predict the next 5 RAW actions.

Output requirements:
- Output JSON only.
- Predict exactly 5 actions.
- Each action must contain exactly three numeric fields: dx, dz, yaw.
- Use RAW actions, not normalized actions.
- Do not output any explanation text.

Primary objectives:
- Reach the goal as accurately as possible.
- Respect the accessibility label semantics.
- Stay collision-free whenever bypass is required.
- After clearing the obstacle, re-align toward the goal instead of keeping unnecessary lateral drift.

Action definition:
- dx: lateral motion in the current robot-local frame. Positive dx means move to the robot's RIGHT. Negative dx means move to the robot's LEFT.
- dz: forward motion in the current robot-local frame.
- yaw: heading change in degrees.

Important interpretation note:
- The first 2 observed actions are only pre-event context.
- They are useful for scale reference, but they are NOT an upper bound on future avoidance magnitude.
- Repeating the exact same action 5 times is usually wrong when obstacle avoidance or goal re-alignment is needed.

Quadruped accessibility label: {label}
Guidance for this label:
{label_guidance}

Observed RAW action summary:
{_prefix_summary(quad["observed_actions"])}
- average |dx| of observed actions: {stats["avg_abs_dx"]:.3f}
- average dz of observed actions: {stats["avg_dz"]:.3f}
- average |yaw| of observed actions: {stats["avg_abs_yaw"]:.3f}
- max |dx| in observed actions: {stats["max_abs_dx"]:.3f}
- max |yaw| in observed actions: {stats["max_abs_yaw"]:.3f}

Useful raw-scale priors for this task family:
- dz is usually smooth and forward-dominant; avoid locking all five steps to the exact same value unless the scene clearly supports that.
- |dx| is often modest but can temporarily exceed the observed |dx| when a bypass is necessary.
- yaw should usually stay smooth and continuous; large integer jumps are suspicious unless the scene clearly requires them.
- Prefer realistic decimal values over coarse quantized templates.

Anti-template rules:
- Avoid integer-locked yaw patterns such as -5, -3, -2, 0, 2, 3, 5 repeated across many scenes.
- Avoid making all five dz values exactly identical.
- Avoid arithmetic ladders like dx = -0.20, -0.15, -0.10, 0.05, 0.10 unless the images and geometry clearly justify that pattern.

Rollout start pose:
- position_world = ({rp[0]:.4f}, {rp[1]:.4f}, {rp[2]:.4f})
- yaw_world_deg = {yaw:.2f}

Goal geometry:
- goal_position = {quad["goal_position_text"]}
- goal_distance_band = {quad["goal_distance_band"]}
- goal_distance_m = {quad["goal_distance_m"]:.2f}
- goal_local = ({quad["goal_local_relative_to_rollout_start"]["dx_local"]:.4f}, {quad["goal_local_relative_to_rollout_start"]["dz_local"]:.4f})
- naive_goal_per_step_local = ({quad["naive_goal_per_step_local"]["dx_local_per_step"]:.4f}, {quad["naive_goal_per_step_local"]["dz_local_per_step"]:.4f})
- goal_lateral_direction_hint = {quad["goal_lateral_direction_hint"]}

Obstacle geometry:
- obstacle_position = {quad["obstacle_position_text"]}
- obstacle_distance_band = {quad["obstacle_distance_band"]}
- obstacle_distance_m = {quad["obstacle_distance_m"]:.2f}
- obstacle_local = ({quad["obstacle_local_relative_to_rollout_start"]["dx_local"]:.4f}, {quad["obstacle_local_relative_to_rollout_start"]["dz_local"]:.4f})
- required_pass_side = {quad["required_pass_side"]}
- obstacle_protected_radius_hint_m = {quad["protected_radius_hint"]:.2f}
- suggested_bypass_corridor_dx_local = {quad["suggested_bypass_corridor_dx_local"]:.4f}
- early_bypass_direction_hint = {quad["early_bypass_direction_hint"]}

Bypass side rule:
- If required_pass_side = left_of_obstacle, then while passing the obstacle the trajectory should stay at a local x that is more LEFT than the obstacle, meaning trajectory x should be meaningfully smaller than obstacle_local.dx_local.
- If required_pass_side = right_of_obstacle, then while passing the obstacle the trajectory should stay at a local x that is more RIGHT than the obstacle, meaning trajectory x should be meaningfully larger than obstacle_local.dx_local.
- The suggested_bypass_corridor_dx_local is a temporary target near obstacle depth, not the final goal.
- early_bypass_direction_hint tells you which sign of dx should usually dominate in the first 1 to 3 actions while establishing safe clearance.

Planning guidance:
- Use the images to judge free space and obstacle appearance, but also respect the provided geometry.
- For Passable, do not insert an unnecessary detour just because the obstacle exists.
- For Bypass, if the obstacle is ahead in local z, start building lateral clearance before reaching obstacle depth.
- First create enough lateral separation for a safe bypass, then recover toward the goal.
- For Blocked, keep the next 5 actions close to zero so the quadruped stops instead of continuing forward.
- Do not over-steer: prefer smooth, realistic corrections over exaggerated yaw jumps.
- If the obstacle is near the centerline and a bypass is required, near-zero dx for all 5 steps is usually wrong.
- Let later actions partially undo the early bypass offset once the obstacle is safely cleared.

Recommended three-phase structure:
- Phase 1: begin moving toward the safe bypass corridor while keeping forward progress.
- Phase 2: pass the obstacle on the required side with enough lateral clearance.
- Phase 3: reduce the remaining lateral offset and re-align toward the goal.

Self-check before answering:
- Do the 5 actions represent a plausible approach, bypass, and re-alignment sequence?
- Is the bypass on the correct side according to the required pass side?
- Are the yaw values smooth and realistic rather than coarse quantized placeholders?
- Are the dz values plausible and not locked to a single copied constant without reason?
- Is there enough lateral deviation to satisfy the bypass when needed?
- After clearing the obstacle, do later actions steer back toward the goal?

Return exactly this JSON structure with numeric values:
{{
  "predicted_actions": [
    {{"dx": NUMBER, "dz": NUMBER, "yaw": NUMBER}},
    {{"dx": NUMBER, "dz": NUMBER, "yaw": NUMBER}},
    {{"dx": NUMBER, "dz": NUMBER, "yaw": NUMBER}},
    {{"dx": NUMBER, "dz": NUMBER, "yaw": NUMBER}},
    {{"dx": NUMBER, "dz": NUMBER, "yaw": NUMBER}}
  ]
}}
""".strip()
