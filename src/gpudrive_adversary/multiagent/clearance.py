"""Signed oriented-footprint clearance for the ten controlled vehicles."""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np


class ClearanceError(ValueError):
    """Raised when vehicle geometry is incomplete or non-finite."""


def _box(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (5,) or not bool(np.isfinite(array).all()):
        raise ClearanceError(f"{name} must be finite [x, y, heading, length, width]")
    if array[3] <= 0 or array[4] <= 0:
        raise ClearanceError(f"{name} dimensions must be positive")
    return array


def oriented_box_corners(box: Any) -> np.ndarray:
    x, y, heading, length, width = _box(box, "box")
    forward = np.asarray([np.cos(heading), np.sin(heading)])
    left = np.asarray([-np.sin(heading), np.cos(heading)])
    center = np.asarray([x, y])
    return np.stack(
        (
            center + forward * length / 2 + left * width / 2,
            center - forward * length / 2 + left * width / 2,
            center - forward * length / 2 - left * width / 2,
            center + forward * length / 2 - left * width / 2,
        )
    )


def _point_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    delta = end - start
    denominator = float(delta @ delta)
    if denominator == 0:
        return float(np.linalg.norm(point - start))
    fraction = float(np.clip(((point - start) @ delta) / denominator, 0.0, 1.0))
    return float(np.linalg.norm(point - (start + fraction * delta)))


def _unsigned_polygon_distance(first: np.ndarray, second: np.ndarray) -> float:
    distances: list[float] = []
    for point in first:
        for index in range(4):
            distances.append(_point_segment_distance(point, second[index], second[(index + 1) % 4]))
    for point in second:
        for index in range(4):
            distances.append(_point_segment_distance(point, first[index], first[(index + 1) % 4]))
    return min(distances)


def oriented_box_clearance(first: Any, second: Any) -> float:
    """Return exact positive separation or negative SAT penetration depth.

    Touching boxes return zero. For overlapping boxes, the magnitude is the
    smallest translation along either box axis that separates the footprints.
    """

    a = _box(first, "first box")
    b = _box(second, "second box")
    corners_a = oriented_box_corners(a)
    corners_b = oriented_box_corners(b)
    axes = []
    for heading in (a[2], b[2]):
        axes.extend(
            (
                np.asarray([np.cos(heading), np.sin(heading)]),
                np.asarray([-np.sin(heading), np.cos(heading)]),
            )
        )
    gaps: list[float] = []
    for axis in axes:
        projection_a = corners_a @ axis
        projection_b = corners_b @ axis
        gaps.append(float(max(projection_a.min() - projection_b.max(), projection_b.min() - projection_a.max())))
    if max(gaps) <= 0.0:
        return max(gaps)
    return _unsigned_polygon_distance(corners_a, corners_b)


def minimum_pairwise_clearance(
    boxes: Any, active: Any | None = None
) -> tuple[float, tuple[int, int]]:
    """Return the minimum signed clearance and stable lowest-index pair."""

    values = np.asarray(boxes, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 5 or not bool(np.isfinite(values).all()):
        raise ClearanceError("boxes must be a finite N-by-5 array")
    if active is None:
        mask = np.ones(values.shape[0], dtype=np.bool_)
    else:
        mask = np.asarray(active, dtype=np.bool_)
        if mask.shape != (values.shape[0],):
            raise ClearanceError("active mask shape does not match boxes")
    indices = np.flatnonzero(mask).tolist()
    if len(indices) < 2:
        return float("inf"), (-1, -1)
    candidates = [
        (oriented_box_clearance(values[first], values[second]), (first, second))
        for first, second in combinations(indices, 2)
    ]
    return min(candidates, key=lambda item: (item[0], item[1]))
