from __future__ import annotations

"""
Entity definitions for the pursuit arena.
"""

from dataclasses import dataclass, field
from typing import List, Tuple

from .geometry import Vec2


@dataclass
class WallStroke:
    """
    A wall drawn by the user as a polyline.

    For collision and visibility we treat consecutive points as line segments.
    """

    points: List[Vec2]
    thickness: int = 6


@dataclass
class PoliceAgent:
    """Police agent trying to arrest enemies."""

    position: Vec2
    direction: float  # radians, 0 = +x axis
    speed: float
    radius: float
    fov_angle: float
    vision_range: float
    arrest_radius: float

    # For manual control or RL we can store additional state here later


@dataclass
class EnemyAgent:
    """Enemy agent trying to escape the map."""

    position: Vec2
    direction: float
    speed: float
    radius: float

    current_target: Tuple[float, float] | None = None
    danger_score: float = 0.0
    last_visible_to_police: bool = False
    blocked_last_step: bool = False  # True when movement was blocked by wall (did not move)


@dataclass
class WorldState:
    """
    Full state of the 2D world at a given time.

    Designed for single police/single enemy but can be extended later.
    """

    width: int
    height: int
    police_agents: List[PoliceAgent] = field(default_factory=list)
    enemy_agents: List[EnemyAgent] = field(default_factory=list)
    walls: List[WallStroke] = field(default_factory=list)

    # Simulation flags
    running: bool = False
    step_count: int = 0

