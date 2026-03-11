from __future__ import annotations

"""
Scripted police controller (e.g. chase toward enemy).
Used when training the enemy agent so police is not RL.
"""

import math
from typing import Tuple

from .entities import EnemyAgent, PoliceAgent
from .geometry import vector_to_angle


def scripted_police_chase(police: PoliceAgent, enemy: EnemyAgent) -> Tuple[float, float]:
    """
    Return (forward_speed, angular_delta) for the police to chase the enemy.
    Turn toward enemy then move forward.
    """
    dx = enemy.position[0] - police.position[0]
    dy = enemy.position[1] - police.position[1]
    if dx == 0 and dy == 0:
        return 0.0, 0.0
    target_angle = math.atan2(dy, dx)
    # Angle difference in [-pi, pi]
    diff = (target_angle - police.direction + math.pi) % (2.0 * math.pi) - math.pi
    turn = max(-0.15, min(0.15, diff))
    forward = police.speed
    return forward, turn
