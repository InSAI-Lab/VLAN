from __future__ import annotations


def binary_score(flag: bool | None, default_when_none: float = 1.0) -> float:
    if flag is None:
        return default_when_none
    return 1.0 if flag else 0.0


def mention_score(mentioned: bool, semantic_match: bool | None) -> float:
    if semantic_match is True:
        return 1.0
    return 0.0


def bounded_error_score(error_value: float | None, tolerance: float, default_when_missing: float = 0.0) -> float:
    if error_value is None:
        return default_when_missing
    return max(0.0, 1.0 - (error_value / tolerance))


def weighted_average(components: dict[str, tuple[float, float]]) -> float:
    total_weight = sum(weight for weight, _ in components.values())
    if total_weight <= 0.0:
        return 0.0
    return sum(weight * score for weight, score in components.values()) / total_weight
