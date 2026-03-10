from __future__ import annotations

"""
Geometry and collision utilities for the pursuit arena.
"""

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple
import math

Vec2 = Tuple[float, float]


def add(a: Vec2, b: Vec2) -> Vec2:
    return a[0] + b[0], a[1] + b[1]


def sub(a: Vec2, b: Vec2) -> Vec2:
    return a[0] - b[0], a[1] - b[1]


def mul(a: Vec2, scalar: float) -> Vec2:
    return a[0] * scalar, a[1] * scalar


def length(v: Vec2) -> float:
    return math.hypot(v[0], v[1])


def normalize(v: Vec2) -> Vec2:
    l = length(v)
    if l == 0:
        return 0.0, 0.0
    return v[0] / l, v[1] / l


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def angle_to_vector(angle_rad: float) -> Vec2:
    return math.cos(angle_rad), math.sin(angle_rad)


def vector_to_angle(v: Vec2) -> float:
    return math.atan2(v[1], v[0])


def distance(a: Vec2, b: Vec2) -> float:
    return length(sub(a, b))


@dataclass
class Segment:
    """Line segment represented by two endpoints."""

    p1: Vec2
    p2: Vec2


def segment_intersection(s1: Segment, s2: Segment) -> Optional[Vec2]:
    """Return the intersection point of two segments if they intersect, else None."""
    (x1, y1), (x2, y2) = s1.p1, s1.p2
    (x3, y3), (x4, y4) = s2.p1, s2.p2

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0:
        return None

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

    if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
        px = x1 + t * (x2 - x1)
        py = y1 + t * (y2 - y1)
        return px, py
    return None


def ray_segment_intersection(origin: Vec2, direction: Vec2, segment: Segment) -> Optional[Vec2]:
    """Return intersection point between a ray (origin + t*direction, t>=0) and segment."""
    (x1, y1) = origin
    dx, dy = direction
    (x3, y3), (x4, y4) = segment.p1, segment.p2

    denom = dx * (y3 - y4) - dy * (x3 - x4)
    if denom == 0:
        return None

    t = ((x3 - x1) * (y3 - y4) - (y3 - y1) * (x3 - x4)) / denom
    u = -((dx) * (y1 - y3) - (dy) * (x1 - x3)) / denom

    if t >= 0.0 and 0.0 <= u <= 1.0:
        return x1 + t * dx, y1 + t * dy
    return None


def point_in_fov(
    origin: Vec2,
    facing_angle: float,
    fov_angle: float,
    max_range: float,
    target: Vec2,
) -> bool:
    """Check if target lies within a field-of-view cone."""
    to_target = sub(target, origin)
    d = length(to_target)
    if d == 0 or d > max_range:
        return False
    dir_vec = angle_to_vector(facing_angle)
    to_target_norm = normalize(to_target)
    dot = dir_vec[0] * to_target_norm[0] + dir_vec[1] * to_target_norm[1]
    # Clamp due to floating point
    dot = clamp(dot, -1.0, 1.0)
    angle = math.degrees(math.acos(dot))
    return angle <= fov_angle / 2.0


def line_of_sight_clear(
    origin: Vec2,
    target: Vec2,
    wall_segments: Iterable[Segment],
) -> bool:
    """Return True if the line between origin and target is not blocked by walls."""
    seg = Segment(origin, target)
    for w in wall_segments:
        if segment_intersection(seg, w) is not None:
            return False
    return True


def clamp_to_bounds(pos: Vec2, width: int, height: int) -> Vec2:
    return clamp(pos[0], 0.0, float(width)), clamp(pos[1], 0.0, float(height))


def distance_to_rect_edges(pos: Vec2, width: int, height: int) -> Tuple[float, float, float, float]:
    """
    Distance from a point to each side of the rectangle (left, right, top, bottom).
    Useful for observations and escape computations.
    """
    x, y = pos
    return x, width - x, y, height - y


def nearest_exit_distance(pos: Vec2, width: int, height: int) -> float:
    """Shortest distance from point to any map boundary."""
    left, right, top, bottom = distance_to_rect_edges(pos, width, height)
    return min(left, right, top, bottom)

