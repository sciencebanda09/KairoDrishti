from __future__ import annotations

from math import exp, hypot

from .models import GeoPoint, SearchCell


def _distance_m(a: GeoPoint, b: GeoPoint) -> float:
    return hypot((a.lat - b.lat) * 111_000, (a.lon - b.lon) * 111_000)


def rank_search_cells(
    cells: list[SearchCell], last_known_position: GeoPoint,
    *, movement_radius_m: float = 1200.0,
) -> list[SearchCell]:
    """Rank cells using LKP distance, movement likelihood, terrain and coverage."""
    for cell in cells:
        distance_score = exp(-(_distance_m(cell.center, last_known_position) /
                               max(movement_radius_m, 1.0)) ** 2)
        unsearched = max(0.0, min(1.0, 1.0 - cell.covered_fraction))
        cell.likelihood = max(0.0, min(1.0,
            0.42 * distance_score + 0.25 * cell.movement_score +
            0.18 * cell.terrain_score + 0.10 * cell.visibility_score +
            0.05 * unsearched))
    ranked = sorted(cells, key=lambda cell: cell.likelihood, reverse=True)
    for priority, cell in enumerate(ranked, 1):
        cell.priority = priority
    return ranked
