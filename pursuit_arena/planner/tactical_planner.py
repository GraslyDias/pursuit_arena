from __future__ import annotations

"""
Simple 2D tactical planner skeleton for future multi-agent coordination.

At this stage we only define data structures and interfaces.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Tuple

from ..core.entities import WorldState


class Role(Enum):
    """Roles that police agents might take in a coordinated plan."""

    CHASER = auto()
    BLOCKER = auto()
    FLANKER = auto()


@dataclass
class BlockingPoint:
    """A point on the map where an agent should attempt to stand to block escape."""

    position: Tuple[float, float]


@dataclass
class TacticalAssignment:
    """Role and target for a single police agent."""

    police_index: int
    role: Role
    target_point: BlockingPoint | None = None


@dataclass
class TacticalPlan:
    """
    High-level plan for all police agents.

    This can later include timing, ordering, and contingencies.
    """

    assignments: List[TacticalAssignment]


class TacticalPlanner:
    """
    Skeleton 2D tactical planner.

    Future responsibilities:
    - detect likely escape direction
    - choose blocking points near exits
    - assign roles (chaser, blocker, flanker)
    - update plans as enemy moves
    """

    def __init__(self) -> None:
        ...

    def propose_plan(self, world: WorldState) -> TacticalPlan:
        """
        Construct a simple placeholder plan.

        For now, we only generate an empty plan; logic will be added later.
        """
        assignments: List[TacticalAssignment] = []
        return TacticalPlan(assignments=assignments)

