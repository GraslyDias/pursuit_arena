from __future__ import annotations

"""
Global configuration values for the pursuit arena simulation.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class WorldConfig:
    """Configuration for the 2D world and agents."""

    width: int = 1280
    height: int = 720

    # Agent defaults
    police_radius: float = 10.0
    enemy_radius: float = 10.0
    police_speed: float = 3.0
    enemy_speed: float = 2.5

    police_fov_deg: float = 90.0
    police_vision_range: float = 250.0
    police_arrest_radius: float = 20.0

    # Simulation
    time_step: float = 1.0  # abstract step duration, not real time


DEFAULT_WORLD_CONFIG = WorldConfig()

