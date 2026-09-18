"""Turn per-label sigmoid probabilities into binary presence/absence calls."""

from __future__ import annotations


def resolve_thresholds(
    label_names: list[str],
    default_threshold: float,
    overrides: dict[str, float] | None = None,
) -> dict[str, float]:
    overrides = overrides or {}
    return {name: overrides.get(name, default_threshold) for name in label_names}


def apply_thresholds(
    probabilities: dict[str, float],
    thresholds: dict[str, float],
) -> dict[str, int]:
    return {
        name: int(prob >= thresholds.get(name, 0.5))
        for name, prob in probabilities.items()
    }
