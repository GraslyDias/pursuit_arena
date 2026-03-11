from __future__ import annotations

"""
World update and high-level simulation logic.
"""

from typing import Iterable, List, Tuple
import math

from .config import DEFAULT_WORLD_CONFIG, WorldConfig
from .entities import EnemyAgent, PoliceAgent, WallStroke, WorldState
from .geometry import (
    Vec2,
    add,
    angle_to_vector,
    clamp_to_bounds,
    distance,
    line_of_sight_clear,
    nearest_exit_distance,
    point_in_fov,
    Segment,
    segment_intersection,
)


def wall_segments_from_strokes(walls: Iterable[WallStroke]) -> List[Segment]:
    segments: List[Segment] = []
    for w in walls:
        if len(w.points) < 2:
            continue
        for i in range(len(w.points) - 1):
            segments.append(Segment(w.points[i], w.points[i + 1]))
    return segments


def _apply_wall_collision(old_pos: Vec2, new_pos: Vec2, wall_segs: List[Segment]) -> Vec2:
    """
    Simple collision: if movement segment intersects any wall, stay at old_pos.

    This is conservative but easy to understand and good enough for a first
    tactical prototype.
    """
    move_seg = Segment(old_pos, new_pos)
    for w in wall_segs:
        if segment_intersection(move_seg, w) is not None:
            return old_pos
    return new_pos


def update_world(
    state: WorldState,
    police_actions: List[Tuple[float, float]] | None = None,
    enemy_dirs: List[float] | None = None,
    config: WorldConfig = DEFAULT_WORLD_CONFIG,
) -> Tuple[bool, dict]:
    """
    Advance the world by one simulation step.

    Parameters
    ----------
    state:
        WorldState to mutate.
    police_actions:
        Optional per-police movement control as (forward_speed, angular_delta).
        If None, police remain with their current velocity direction.
    enemy_dirs:
        Optional per-enemy absolute movement direction angles in radians.
        If None, enemies remain pointing in their current direction.
    config:
        World configuration.

    Returns
    -------
    terminated, info
        terminated is True if a terminal condition (arrest/escape) happened.
        info contains details: {"arrested": bool, "escaped": bool}.
    """
    width, height = state.width, state.height
    wall_segs = wall_segments_from_strokes(state.walls)

    # --- Update police ---
    for idx, police in enumerate(state.police_agents):
        forward_speed = config.police_speed
        angular_delta = 0.0
        if police_actions is not None and idx < len(police_actions):
            forward_speed, angular_delta = police_actions[idx]

        police.direction += angular_delta
        move_vec = angle_to_vector(police.direction)
        old_pos = police.position
        desired_pos = (
            old_pos[0] + move_vec[0] * forward_speed,
            old_pos[1] + move_vec[1] * forward_speed,
        )
        # Clamp to world bounds, then apply wall collision
        desired_pos = clamp_to_bounds(desired_pos, width, height)
        new_pos = _apply_wall_collision(old_pos, desired_pos, wall_segs)
        police.position = new_pos
        # If blocked by wall (barely moved), nudge rotation so police can turn and get unstuck
        if distance(old_pos, new_pos) < 0.5:
            police.direction += 0.12  # ~7° per frame when stuck; next step may succeed

    # --- Update enemies ---
    # Enemy is NOT clamped to map: they can move outside the screen to escape.
    # Only walls block them. Police stay inside (clamped above).
    for idx, enemy in enumerate(state.enemy_agents):
        if enemy_dirs is not None and idx < len(enemy_dirs):
            enemy.direction = enemy_dirs[idx]
        move_vec = angle_to_vector(enemy.direction)
        old_pos = enemy.position
        desired_pos = (
            old_pos[0] + move_vec[0] * enemy.speed,
            old_pos[1] + move_vec[1] * enemy.speed,
        )
        # Do NOT clamp enemy to bounds — leaving the map counts as escape
        new_pos = _apply_wall_collision(old_pos, desired_pos, wall_segs)
        enemy.position = new_pos
        # Stuck = still inside map and barely moved (blocked by wall)
        still_inside = (0 <= new_pos[0] <= width and 0 <= new_pos[1] <= height)
        enemy.blocked_last_step = still_inside and (distance(old_pos, new_pos) < 0.5)

    state.step_count += 1

    # --- Arrest & escape detection ---
    arrested_any = False
    escaped_any = False

    for enemy in state.enemy_agents:
        # Escape: leaving the map
        if (
            enemy.position[0] < 0
            or enemy.position[0] > width
            or enemy.position[1] < 0
            or enemy.position[1] > height
        ):
            escaped_any = True
            continue

        # Arrest: any police within arrest radius and line of sight
        for police in state.police_agents:
            d = distance(police.position, enemy.position)
            if d <= police.arrest_radius:
                # No need for line-of-sight if within radius (close contact)
                arrested_any = True
                break
        if arrested_any:
            break

    terminated = arrested_any or escaped_any
    info = {"arrested": arrested_any, "escaped": escaped_any}
    return terminated, info


def police_can_see_enemy(
    police: PoliceAgent,
    enemy: EnemyAgent,
    walls: Iterable[WallStroke],
) -> bool:
    """Check if a given police agent can see an enemy."""
    wall_segs = wall_segments_from_strokes(walls)
    in_fov = point_in_fov(
        police.position,
        police.direction,
        police.fov_angle,
        police.vision_range,
        enemy.position,
    )
    if not in_fov:
        return False
    return line_of_sight_clear(police.position, enemy.position, wall_segs)


def compute_enemy_visibility_and_danger(
    state: WorldState, config: WorldConfig = DEFAULT_WORLD_CONFIG
) -> None:
    """
    Update enemy fields: last_visible_to_police and danger_score.

    Danger is a simple heuristic based on:
    - distance to nearest police
    - whether visible to any police
    - distance to nearest exit (farther from exit is more dangerous)
    """
    wall_segs = wall_segments_from_strokes(state.walls)
    for enemy in state.enemy_agents:
        min_police_dist = math.inf
        visible_any = False

        for police in state.police_agents:
            d = distance(police.position, enemy.position)
            min_police_dist = min(min_police_dist, d)

            in_fov = point_in_fov(
                police.position,
                police.direction,
                police.fov_angle,
                police.vision_range,
                enemy.position,
            )
            if in_fov and line_of_sight_clear(police.position, enemy.position, wall_segs):
                visible_any = True

        enemy.last_visible_to_police = visible_any

        # Simple danger heuristic: closer police and farther from exit is more dangerous
        dist_exit = nearest_exit_distance(enemy.position, state.width, state.height)
        inv_police_dist = 0.0 if min_police_dist == math.inf else 1.0 / max(min_police_dist, 1.0)
        enemy.danger_score = inv_police_dist + (dist_exit / max(state.width, state.height))

