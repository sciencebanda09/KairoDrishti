"""Canonical terrain-search detector taxonomy."""
from __future__ import annotations

TERRAIN_CLASSES = ("person", "clothing", "shelter", "track")


def validate_class_names(names: list[str] | tuple[str, ...] | dict[int, str]) -> tuple[str, ...]:
    values = tuple(names.values()) if isinstance(names, dict) else tuple(names)
    normalized = tuple(str(value).strip().lower() for value in values)
    missing = [label for label in TERRAIN_CLASSES if label not in normalized]
    if missing:
        raise ValueError(f"terrain detector is missing required classes: {', '.join(missing)}")
    return normalized
