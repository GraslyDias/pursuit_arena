from __future__ import annotations

"""
Scripted enemy controller for escape behavior.
"""

from typing import Iterable, List
import math
import random

from .config import DEFAULT_WORLD_CONFIG, WorldConfig
from .entities import EnemyAgent, PoliceAgent, WallStroke, WorldState
from .geometry import (
    Vec2,
    add,
    angle_to_vector,
    distance,
    nearest_exit_distance,
    normalize,
    ray_segment_intersection,
    Segment,
)
from .world import wall_segments_from_strokes, police_can_see_enemy


def _score_direction(
    direction: float,
    enemy: EnemyAgent,
    police_agents: List[PoliceAgent],
    wall_segments: List[Segment],
    world_width: int,
    world_height: int,
    config: WorldConfig,
) -> float:
    """
    Score a candidate movement direction for the enemy.

    Higher is better. Combines:
    - progress toward nearest map exit
    - penalty for wall collision risk
    - penalty for being close to police
    - penalty for being in police FOV
    - bonus if walls block line of sight
    """
    pos = enemy.position
    move_vec = angle_to_vector(direction)
    step_pos = add(pos, (move_vec[0] * enemy.speed, move_vec[1] * enemy.speed))

    # 1) Progress toward nearest exit
    dist_now = nearest_exit_distance(pos, world_width, world_height)
    dist_future = nearest_exit_distance(step_pos, world_width, world_height)
    progress = dist_now - dist_future  # positive if moving closer

    # 2) Wall collision risk: ray in movement direction, penalize if close hit
    ray = Segment(pos, step_pos)
    wall_penalty = 0.0
    for seg in wall_segments:
        hit = ray_segment_intersection(ray.p1, (move_vec[0], move_vec[1]), seg)
        if hit is not None:
            d = distance(pos, hit)
            wall_penalty += max(0.0, 1.0 - d / 30.0)

    # 3) Police proximity and visibility
    min_police_dist = math.inf
    fov_penalty = 0.0
    cover_bonus = 0.0

    for police in police_agents:
        d = distance(step_pos, police.position)
        min_police_dist = min(min_police_dist, d)

        # Estimate visibility from future position (approximate)
        tmp_enemy = EnemyAgent(position=step_pos, direction=enemy.direction, speed=enemy.speed, radius=enemy.radius)
        if police_can_see_enemy(police, tmp_enemy, []):  # ignore walls for raw FOV check
            fov_penalty += 1.0

        # Check if wall could block LOS along this direction
        for seg in wall_segments:
            hit = ray_segment_intersection(police.position, normalize((step_pos[0] - police.position[0], step_pos[1] - police.position[1])), seg)
            if hit is not None:
                cover_bonus += 0.5
                break

    police_penalty = 0.0
    if min_police_dist < math.inf:
        police_penalty = max(0.0, 1.0 - min_police_dist / 200.0)

    score = (
        3.0 * progress
        - 4.0 * wall_penalty
        - 5.0 * police_penalty
        - 2.0 * fov_penalty
        + 1.5 * cover_bonus
    )
    return score


def _angle_diff(a: float, b: float) -> float:
    """Smallest difference between two angles in [-pi, pi]."""
    d = (a - b) % (2.0 * math.pi)
    if d > math.pi:
        d -= 2.0 * math.pi
    return d


def choose_enemy_directions(
    state: WorldState,
    num_samples: int = 24,
    config: WorldConfig = DEFAULT_WORLD_CONFIG,
) -> List[float]:
    """
    Compute movement direction (angle in radians) for each enemy using sampling.

    When the enemy was blocked last step (hit a wall), we sample a full 360°
    and penalize the current direction so it rotates and finds a way around.
    """
    wall_segs = wall_segments_from_strokes(state.walls)
    directions: List[float] = []

    for enemy in state.enemy_agents:
        best_score = -math.inf
        best_dir = enemy.direction

        if enemy.blocked_last_step:
            # Stuck at wall: sample full circle so we can turn around / try sides
            for i in range(num_samples):
                cand_dir = (i / num_samples) * 2.0 * math.pi - math.pi
                score = _score_direction(
                    cand_dir,
                    enemy,
                    state.police_agents,
                    wall_segs,
                    state.width,
                    state.height,
                    config,
                )
                # Strong penalty for going back in the same blocked direction
                angle_diff = abs(_angle_diff(cand_dir, enemy.direction))
                if angle_diff < math.radians(45):
                    score -= 8.0
                if score > best_score:
                    best_score = score
                    best_dir = cand_dir
        else:
            # Normal: sample fan in front (±90°)
            for i in range(num_samples):
                offset = (i / max(1, num_samples - 1) - 0.5) * math.pi
                cand_dir = enemy.direction + offset
                score = _score_direction(
                    cand_dir,
                    enemy,
                    state.police_agents,
                    wall_segs,
                    state.width,
                    state.height,
                    config,
                )
                if score > best_score:
                    best_score = score
                    best_dir = cand_dir

        best_dir += random.uniform(-0.08, 0.08)
        directions.append(best_dir)

    return directions

