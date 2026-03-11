from __future__ import annotations

"""
Gymnasium-compatible environment for a single police agent chasing a scripted enemy.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import json
import math
import random

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from ...core.config import DEFAULT_WORLD_CONFIG, WorldConfig
from ...core.entities import EnemyAgent, PoliceAgent, WallStroke, WorldState
from ...core.enemy_ai import choose_enemy_directions
from ...core.geometry import (
    Vec2,
    add,
    angle_to_vector,
    clamp_to_bounds,
    distance,
    length,
    mul,
    nearest_exit_distance,
    normalize,
    sub,
    distance_to_rect_edges,
    vector_to_angle,
)
from ...core.police_ai import scripted_police_chase
from ...core.world import compute_enemy_visibility_and_danger, police_can_see_enemy, update_world


class ChaseEscapeEnv(gym.Env):
    """
    Single-police vs single-enemy environment.
    """

    metadata = {"render_modes": ["human", "none"], "render_fps": 60}

    def __init__(
        self,
        config: WorldConfig | None = None,
        max_steps: int = 600,
        render_mode: str | None = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.config = config or DEFAULT_WORLD_CONFIG
        self.max_steps = max_steps
        self.render_mode = render_mode

        # Action space: discrete movement/rotation commands for the police
        self.action_space = spaces.Discrete(6)

        # Observation space: compact float vector
        # [police_x, police_y, cos(theta), sin(theta),
        #  enemy_visible_flag,
        #  rel_enemy_x, rel_enemy_y, dist_enemy,
        #  dist_nearest_exit,
        #  wall_rays(4),
        #  dist_edges(4),
        #  remaining_time]
        obs_dim = 4 + 1 + 3 + 1 + 4 + 4 + 1
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)

        self._rng = random.Random(seed)
        self.state: Optional[WorldState] = None
        self.steps: int = 0

        # Lazy Pygame renderer
        self._pygame_screen = None
        self._pygame_clock = None

    def seed(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Map / wall generation
    # ------------------------------------------------------------------
    def _generate_random_walls(
        self,
        state: WorldState,
        num_strokes: int | None = None,
        stroke_length: float = 220.0,
    ) -> None:
        """
        Create different random wall layouts each episode for training.

        Uses a random number of strokes and sometimes multi-segment (L-shaped)
        walls so the enemy must rotate and find ways around obstacles.
        """
        w, h = state.width, state.height
        margin = 60.0
        if num_strokes is None:
            num_strokes = self._rng.randint(4, 9)

        for _ in range(num_strokes):
            # Random start point
            x0 = self._rng.uniform(margin, w - margin)
            y0 = self._rng.uniform(margin, h - margin)
            length = self._rng.uniform(stroke_length * 0.5, stroke_length * 1.2)

            if self._rng.random() < 0.35:
                # Multi-segment (L or short polyline) so enemy must navigate around corners
                points = [(x0, y0)]
                for _ in range(2):
                    angle = self._rng.uniform(-math.pi, math.pi)
                    seg_len = length * self._rng.uniform(0.3, 0.7)
                    x1 = max(0.0, min(w, points[-1][0] + math.cos(angle) * seg_len))
                    y1 = max(0.0, min(h, points[-1][1] + math.sin(angle) * seg_len))
                    points.append((x1, y1))
                state.walls.append(WallStroke(points=points, thickness=6))
            else:
                # Single segment
                angle = self._rng.uniform(-math.pi, math.pi)
                x1 = max(0.0, min(w, x0 + math.cos(angle) * length))
                y1 = max(0.0, min(h, y0 + math.sin(angle) * length))
                state.walls.append(WallStroke(points=[(x0, y0), (x1, y1)], thickness=6))

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is not None:
            self.seed(seed)
        # Use saved training map if no options given (e.g. from play_model "Save map")
        if options is None and getattr(self, "training_map_options", None) is not None:
            options = self.training_map_options

        cfg = self.config
        self.state = WorldState(width=cfg.width, height=cfg.height)
        self.state.police_agents.clear()
        self.state.enemy_agents.clear()
        self.state.walls.clear()
        self.steps = 0

        # Custom layout from options (e.g. from play_model Edit mode), or random
        opts = options or {}
        margin = 100.0
        w, h = cfg.width, cfg.height

        if opts.get("walls") is not None:
            for w in opts["walls"]:
                if isinstance(w, WallStroke):
                    self.state.walls.append(w)
                else:
                    points = [tuple(p) for p in w.get("points", [])]
                    self.state.walls.append(WallStroke(points=points, thickness=int(w.get("thickness", 6))))
        else:
            self._generate_random_walls(self.state)

        if opts.get("police_pos") is not None and opts.get("enemy_pos") is not None:
            px, py = opts["police_pos"]
            ex, ey = opts["enemy_pos"]
            police_dir = opts.get("police_dir", 0.0)
            enemy_dir = opts.get("enemy_dir", math.pi)
        else:
            px = self._rng.uniform(margin, w - margin)
            py = self._rng.uniform(margin, h - margin)
            ex = self._rng.uniform(margin, w - margin)
            ey = self._rng.uniform(margin, h - margin)
            police_dir = self._rng.uniform(-math.pi, math.pi)
            enemy_dir = self._rng.uniform(-math.pi, math.pi)

        # Step 1 training: static_enemy=True so enemy never moves; only police is trained to find and arrest
        static_enemy = opts.get("static_enemy") or getattr(self, "static_enemy", False)
        enemy_speed = 0.0 if static_enemy else cfg.enemy_speed

        police = PoliceAgent(
            position=(float(px), float(py)),
            direction=float(police_dir),
            speed=cfg.police_speed,
            radius=cfg.police_radius,
            fov_angle=cfg.police_fov_deg,
            vision_range=cfg.police_vision_range,
            arrest_radius=cfg.police_arrest_radius,
        )
        enemy = EnemyAgent(
            position=(float(ex), float(ey)),
            direction=float(enemy_dir),
            speed=enemy_speed,
            radius=cfg.enemy_radius,
        )
        self.state.police_agents.append(police)
        self.state.enemy_agents.append(enemy)

        compute_enemy_visibility_and_danger(self.state, self.config)
        obs = self._get_obs()
        info: Dict[str, Any] = {}
        return obs, info

    def step(self, action: int):
        assert self.state is not None

        cfg = self.config
        self.steps += 1

        police = self.state.police_agents[0]
        enemy = self.state.enemy_agents[0]
        old_police_pos = police.position

        # Decode discrete action to (forward_speed, angular_delta)
        forward = 0.0
        turn = 0.0
        if action == 1:  # forward
            forward = cfg.police_speed
        elif action == 2:  # rotate left
            turn = -0.1
        elif action == 3:  # rotate right
            turn = 0.1
        elif action == 4:  # forward-left
            forward = cfg.police_speed
            turn = -0.1
        elif action == 5:  # forward-right
            forward = cfg.police_speed
            turn = 0.1

        police_actions = [(forward, turn)]

        # Scripted enemy direction
        enemy_dirs = choose_enemy_directions(self.state, num_samples=16, config=self.config)

        terminated, info = update_world(
            self.state,
            police_actions=police_actions,
            enemy_dirs=enemy_dirs,
            config=self.config,
        )
        compute_enemy_visibility_and_danger(self.state, self.config)

        arrested = info.get("arrested", False)
        escaped = info.get("escaped", False)

        reward = 0.0

        # Police duty: arrest enemy. Enemy goal is to escape (exit screen).
        # Small reward when police detects enemy (enemy inside FOV triangle / line-of-sight)
        enemy_visible = police_can_see_enemy(police, enemy, self.state.walls)
        if enemy_visible:
            reward += 0.05  # small reward for keeping enemy in detection (triangle)

        # Large reward for arrest; large negative when enemy escapes
        if arrested:
            reward += 5.0   # police gets +5 for arresting
        if escaped:
            reward -= 10.0  # police gets -10 when enemy escapes (enemy achieved goal)

        # Negative reward for colliding with walls (police tries to move but stays almost in place).
        moved_police = distance(old_police_pos, police.position)
        inside = 0.0 <= police.position[0] <= cfg.width and 0.0 <= police.position[1] <= cfg.height
        if forward > 0.0 and moved_police < 0.5 and inside and not (arrested or escaped):
            reward -= 0.05

        # Small step penalty to encourage efficiency
        reward -= 0.002

        truncated = self.steps >= self.max_steps
        if truncated and not (arrested or escaped):
            reward -= 1.0

        obs = self._get_obs()

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, {"arrested": arrested, "escaped": escaped}

    def _get_obs(self) -> np.ndarray:
        assert self.state is not None
        cfg = self.config
        police = self.state.police_agents[0]
        enemy = self.state.enemy_agents[0]

        # Normalized positions
        px = police.position[0] / cfg.width
        py = police.position[1] / cfg.height
        dir_cos = math.cos(police.direction)
        dir_sin = math.sin(police.direction)

        # Enemy visibility and relative data
        visible = police_can_see_enemy(police, enemy, self.state.walls)
        enemy_visible_flag = 1.0 if visible else 0.0
        if visible:
            rel_x = (enemy.position[0] - police.position[0]) / cfg.width
            rel_y = (enemy.position[1] - police.position[1]) / cfg.height
            dist_e = distance(police.position, enemy.position) / math.hypot(cfg.width, cfg.height)
        else:
            rel_x = 0.0
            rel_y = 0.0
            dist_e = 0.0

        # Nearest exit distance
        dist_exit = nearest_exit_distance(police.position, cfg.width, cfg.height) / max(cfg.width, cfg.height)

        # Simple wall rays: just distances to boundaries along 4 cardinal directions
        left, right, top, bottom = distance_to_rect_edges(police.position, cfg.width, cfg.height)
        wall_rays = [
            left / cfg.width,
            right / cfg.width,
            top / cfg.height,
            bottom / cfg.height,
        ]

        # Distance to map edges (same as above here)
        edge_dists = wall_rays

        remaining_time = 1.0 - (self.steps / max(1, self.max_steps))

        obs = np.array(
            [
                px,
                py,
                dir_cos,
                dir_sin,
                enemy_visible_flag,
                rel_x,
                rel_y,
                dist_e,
                dist_exit,
                *wall_rays,
                *edge_dists,
                remaining_time,
            ],
            dtype=np.float32,
        )
        return obs

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def render(self) -> None:
        if self.render_mode != "human":
            return
        try:
            import pygame
        except ImportError:
            return

        cfg = self.config
        if self._pygame_screen is None:
            pygame.init()
            self._pygame_screen = pygame.display.set_mode((cfg.width, cfg.height))
            pygame.display.set_caption("ChaseEscapeEnv")
            self._pygame_clock = pygame.time.Clock()

        assert self.state is not None
        screen = self._pygame_screen
        screen.fill((255, 255, 255))

        # Escape zone border on all 4 edges
        escape_band = 12
        escape_surf = pygame.Surface((cfg.width, cfg.height), pygame.SRCALPHA)
        escape_surf.fill((100, 255, 100, 90), (0, 0, cfg.width, escape_band))
        escape_surf.fill((100, 255, 100, 90), (0, cfg.height - escape_band, cfg.width, escape_band))
        escape_surf.fill((100, 255, 100, 90), (0, 0, escape_band, cfg.height))
        escape_surf.fill((100, 255, 100, 90), (cfg.width - escape_band, 0, escape_band, cfg.height))
        screen.blit(escape_surf, (0, 0))

        # Semi-transparent FOV overlay
        fov_surface = pygame.Surface((cfg.width, cfg.height), pygame.SRCALPHA)

        # Draw walls
        for wall in self.state.walls:
            if len(wall.points) >= 2:
                pygame.draw.lines(screen, (0, 0, 0), False, wall.points, wall.thickness)

        police = self.state.police_agents[0]
        enemy = self.state.enemy_agents[0]

        # FOV triangle (yellow, low opacity)
        half_fov_rad = math.radians(police.fov_angle / 2.0)
        left_dir = police.direction - half_fov_rad
        right_dir = police.direction + half_fov_rad
        p0 = police.position
        p1 = (
            p0[0] + math.cos(left_dir) * police.vision_range,
            p0[1] + math.sin(left_dir) * police.vision_range,
        )
        p2 = (
            p0[0] + math.cos(right_dir) * police.vision_range,
            p0[1] + math.sin(right_dir) * police.vision_range,
        )
        pygame.draw.polygon(fov_surface, (255, 255, 0, 80), [p0, p1, p2])

        # Draw agents
        pygame.draw.circle(screen, (0, 0, 255), (int(police.position[0]), int(police.position[1])), int(police.radius))
        pygame.draw.circle(screen, (255, 0, 0), (int(enemy.position[0]), int(enemy.position[1])), int(enemy.radius))

        # Blit FOV overlay
        screen.blit(fov_surface, (0, 0))

        pygame.display.flip()
        if self._pygame_clock is not None:
            self._pygame_clock.tick(self.metadata["render_fps"])

    def close(self) -> None:
        if self._pygame_screen is not None:
            try:
                import pygame

                pygame.quit()
            except ImportError:
                pass
            self._pygame_screen = None


# -------------------------------------------------------------------------
# Helper: police observation (for trained police model in enemy env and dual run)
# -------------------------------------------------------------------------

def get_police_obs(
    state: WorldState,
    config: WorldConfig,
    steps: int = 0,
    max_steps: int = 600,
) -> np.ndarray:
    """Build the same observation vector the police agent sees (for inference with trained model)."""
    police = state.police_agents[0]
    enemy = state.enemy_agents[0]
    w, h = config.width, config.height
    px = police.position[0] / w
    py = police.position[1] / h
    dir_cos = math.cos(police.direction)
    dir_sin = math.sin(police.direction)
    visible = police_can_see_enemy(police, enemy, state.walls)
    enemy_visible_flag = 1.0 if visible else 0.0
    if visible:
        rel_x = (enemy.position[0] - police.position[0]) / w
        rel_y = (enemy.position[1] - police.position[1]) / h
        dist_e = distance(police.position, enemy.position) / math.hypot(w, h)
    else:
        rel_x = rel_y = dist_e = 0.0
    dist_exit = nearest_exit_distance(police.position, w, h) / max(w, h)
    left, right, top, bottom = distance_to_rect_edges(police.position, w, h)
    wall_rays = [left / w, right / w, top / h, bottom / h]
    remaining_time = 1.0 - (steps / max(1, max_steps))
    return np.array(
        [px, py, dir_cos, dir_sin, enemy_visible_flag, rel_x, rel_y, dist_e, dist_exit, *wall_rays, *wall_rays, remaining_time],
        dtype=np.float32,
    )


def get_enemy_obs(
    state: WorldState,
    config: WorldConfig,
    steps: int = 0,
    max_steps: int = 600,
) -> np.ndarray:
    """Build the same observation vector the enemy agent sees (for inference with trained model)."""
    police = state.police_agents[0]
    enemy = state.enemy_agents[0]
    w, h = config.width, config.height
    ex = enemy.position[0] / w
    ey = enemy.position[1] / h
    dc = math.cos(enemy.direction)
    ds = math.sin(enemy.direction)
    rel_px = (police.position[0] - enemy.position[0]) / w
    rel_py = (police.position[1] - enemy.position[1]) / h
    dist_p = distance(police.position, enemy.position) / math.hypot(w, h)
    dist_exit = nearest_exit_distance(enemy.position, w, h) / max(w, h)
    left, right, top, bottom = distance_to_rect_edges(enemy.position, w, h)
    wall_rays = [left / w, right / w, top / h, bottom / h]
    t_rem = 1.0 - (steps / max(1, max_steps))
    return np.array(
        [ex, ey, dc, ds, rel_px, rel_py, dist_p, dist_exit, *wall_rays, t_rem],
        dtype=np.float32,
    )


# -------------------------------------------------------------------------
# Strategy planner env: 3rd agent – coordinates police (trap, bottleneck, block exit)
# Team shared rewards; supports 1+ police, 1+ enemy (1v1 for now).
# -------------------------------------------------------------------------

def get_strategy_obs(
    state: WorldState,
    config: WorldConfig,
    steps: int = 0,
    max_steps: int = 600,
) -> np.ndarray:
    """Global state for strategy planner: police, enemy, exits, team view."""
    w, h = config.width, config.height
    police = state.police_agents[0]
    enemy = state.enemy_agents[0]
    # Normalized positions
    px = police.position[0] / w
    py = police.position[1] / h
    ex = enemy.position[0] / w
    ey = enemy.position[1] / h
    # Police direction
    dir_cos = math.cos(police.direction)
    dir_sin = math.sin(police.direction)
    # Distances
    d_police_enemy = distance(police.position, enemy.position) / math.hypot(w, h)
    d_enemy_exit = nearest_exit_distance(enemy.position, w, h) / max(w, h)
    d_police_exit = nearest_exit_distance(police.position, w, h) / max(w, h)
    # Which exit is nearest to enemy (left/right/top/bottom)
    left, right, top, bottom = distance_to_rect_edges(enemy.position, w, h)
    nearest = min((left, 0), (right, 1), (top, 2), (bottom, 3), key=lambda x: x[0])
    exit_side = nearest[1] / 3.0  # 0..1
    # Police edge distances
    pleft, pright, ptop, pbottom = distance_to_rect_edges(police.position, w, h)
    wall_rays = [pleft / w, pright / w, ptop / h, pbottom / h]
    t_rem = 1.0 - (steps / max(1, max_steps))
    return np.array(
        [px, py, ex, ey, dir_cos, dir_sin, d_police_enemy, d_enemy_exit, d_police_exit, exit_side, *wall_rays, t_rem],
        dtype=np.float32,
    )


def strategy_action_to_police(state: WorldState, config: WorldConfig, action: int) -> Tuple[float, float]:
    """
    Map strategy discrete action to (forward_speed, angular_delta) for the single police.
    0=chase enemy, 1=block nearest exit, 2=bottleneck (between enemy and exit), 3=hold, 4=flank left, 5=flank right.
    """
    police = state.police_agents[0]
    enemy = state.enemy_agents[0]
    w, h = config.width, config.height
    speed = config.police_speed
    left, right, top, bottom = distance_to_rect_edges(enemy.position, w, h)
    # Nearest exit point on boundary (center of that edge)
    if left <= min(right, top, bottom):
        exit_pt: Vec2 = (0.0, enemy.position[1])
    elif right <= min(left, top, bottom):
        exit_pt = (float(w), enemy.position[1])
    elif top <= min(left, right, bottom):
        exit_pt = (enemy.position[0], 0.0)
    else:
        exit_pt = (enemy.position[0], float(h))
    # Bottleneck: midpoint between enemy and exit_pt
    bottleneck = ((enemy.position[0] + exit_pt[0]) / 2, (enemy.position[1] + exit_pt[1]) / 2)
    if action == 0:  # chase
        target = enemy.position
    elif action == 1:  # block exit
        target = exit_pt
    elif action == 2:  # bottleneck
        target = bottleneck
    elif action == 3:  # hold
        return 0.0, 0.0
    elif action == 4:  # flank left (perpendicular left of enemy-to-police)
        to_p = sub(police.position, enemy.position)
        perp = (-to_p[1], to_p[0])
        perp_n = normalize(perp) if length(perp) > 0 else (1, 0)
        target = add(enemy.position, mul(perp_n, 80))
    elif action == 5:  # flank right
        to_p = sub(police.position, enemy.position)
        perp = (to_p[1], -to_p[0])
        perp_n = normalize(perp) if length(perp) > 0 else (1, 0)
        target = add(enemy.position, mul(perp_n, 80))
    else:
        target = enemy.position
    to_target = sub(target, police.position)
    if length(to_target) < 1.0:
        return 0.0, 0.0
    target_angle = vector_to_angle(to_target)
    diff = (target_angle - police.direction + math.pi) % (2.0 * math.pi) - math.pi
    turn = max(-0.15, min(0.15, diff))
    return speed, turn


class ChaseEscapeStrategyEnv(gym.Env):
    """
    Strategy planner is the RL agent. Commands police (chase, block exit, bottleneck, hold, flank).
    Team shared reward: + for arrest, - for escape. Enemy is scripted or loaded model.
    Saves to ppo_strategy_final.zip (3rd model).
    """

    metadata = {"render_modes": ["human", "none"], "render_fps": 60}

    def __init__(
        self,
        config: WorldConfig | None = None,
        max_steps: int = 600,
        render_mode: str | None = None,
        seed: Optional[int] = None,
        enemy_model: Optional[Any] = None,
    ) -> None:
        super().__init__()
        self.config = config or DEFAULT_WORLD_CONFIG
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.enemy_model = enemy_model
        self.action_space = spaces.Discrete(6)  # chase, block, bottleneck, hold, flank_l, flank_r
        # strategy obs: px,py,ex,ey, dir_cos,dir_sin, d_pe, d_ee, d_pe_exit, exit_side, 4 wall_rays, t_rem
        obs_dim = 4 + 2 + 3 + 1 + 4 + 1
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        self._rng = random.Random(seed)
        self.state: Optional[WorldState] = None
        self.steps: int = 0
        self._pygame_screen = None
        self._pygame_clock = None
        self.last_strategy_action: Optional[int] = None

    def seed(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)

    def _generate_random_walls(self, state: WorldState) -> None:
        _tmp = ChaseEscapeEnv(self.config, self.max_steps, None, None)
        _tmp._rng = self._rng
        _tmp._generate_random_walls(state)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is not None:
            self.seed(seed)
        if options is None and getattr(self, "training_map_options", None) is not None:
            options = self.training_map_options
        cfg = self.config
        self.state = WorldState(width=cfg.width, height=cfg.height)
        self.state.police_agents.clear()
        self.state.enemy_agents.clear()
        self.state.walls.clear()
        self.steps = 0
        opts = options or {}
        margin = 100.0
        w, h = cfg.width, cfg.height
        if opts.get("walls") is not None:
            for wb in opts["walls"]:
                if isinstance(wb, WallStroke):
                    self.state.walls.append(wb)
                else:
                    pts = [tuple(p) for p in wb.get("points", [])]
                    self.state.walls.append(WallStroke(points=pts, thickness=int(wb.get("thickness", 6))))
        else:
            self._generate_random_walls(self.state)
        if opts.get("police_pos") is not None and opts.get("enemy_pos") is not None:
            px, py = opts["police_pos"]
            ex, ey = opts["enemy_pos"]
            police_dir = opts.get("police_dir", 0.0)
            enemy_dir = opts.get("enemy_dir", math.pi)
        else:
            px = self._rng.uniform(margin, w - margin)
            py = self._rng.uniform(margin, h - margin)
            ex = self._rng.uniform(margin, w - margin)
            ey = self._rng.uniform(margin, h - margin)
            police_dir = self._rng.uniform(-math.pi, math.pi)
            enemy_dir = self._rng.uniform(-math.pi, math.pi)
        self.state.police_agents.append(PoliceAgent(
            position=(float(px), float(py)), direction=float(police_dir),
            speed=cfg.police_speed, radius=cfg.police_radius,
            fov_angle=cfg.police_fov_deg, vision_range=cfg.police_vision_range,
            arrest_radius=cfg.police_arrest_radius,
        ))
        self.state.enemy_agents.append(EnemyAgent(
            position=(float(ex), float(ey)), direction=float(enemy_dir),
            speed=cfg.enemy_speed, radius=cfg.enemy_radius,
        ))
        compute_enemy_visibility_and_danger(self.state, self.config)
        return get_strategy_obs(self.state, cfg, self.steps, self.max_steps), {}

    def _enemy_direction(self) -> float:
        if self.enemy_model is not None and self.state is not None:
            eo = get_enemy_obs(self.state, self.config, self.steps, self.max_steps)
            ea, _ = self.enemy_model.predict(eo, deterministic=True)
            turn = {0: 0.0, 1: 0.0, 2: -0.12, 3: 0.12, 4: -0.12, 5: 0.12}.get(int(ea), 0.0)
            return self.state.enemy_agents[0].direction + turn
        return choose_enemy_directions(self.state, config=self.config)[0]

    def step(self, action: int):
        assert self.state is not None
        cfg = self.config
        self.steps += 1
        police = self.state.police_agents[0]
        old_police_pos = police.position
        a = int(action)
        self.last_strategy_action = a
        police_actions = [strategy_action_to_police(self.state, cfg, a)]
        enemy_dirs = [self._enemy_direction()]
        terminated, info = update_world(
            self.state, police_actions=police_actions, enemy_dirs=enemy_dirs, config=cfg,
        )
        compute_enemy_visibility_and_danger(self.state, cfg)
        arrested = info.get("arrested", False)
        escaped = info.get("escaped", False)
        reward = 0.0
        if arrested:
            reward += 5.0
        if escaped:
            reward -= 10.0
        # Penalize strategy when police is effectively colliding with walls (tries to move but is stuck),
        # except when action is explicit HOLD (3).
        moved_police = distance(old_police_pos, self.state.police_agents[0].position)
        inside = 0.0 <= self.state.police_agents[0].position[0] <= cfg.width and 0.0 <= self.state.police_agents[0].position[1] <= cfg.height
        if a != 3 and moved_police < 0.5 and inside and not (arrested or escaped):
            reward -= 0.05
        reward -= 0.002
        truncated = self.steps >= self.max_steps
        if truncated and not (arrested or escaped):
            reward -= 0.5
        obs = get_strategy_obs(self.state, cfg, self.steps, self.max_steps)
        return obs, reward, terminated, truncated, {"arrested": arrested, "escaped": escaped}

    def render(self) -> None:
        if self.render_mode != "human" or self.state is None:
            return
        try:
            import pygame
        except ImportError:
            return
        cfg = self.config
        if self._pygame_screen is None:
            pygame.init()
            self._pygame_screen = pygame.display.set_mode((cfg.width, cfg.height))
            pygame.display.set_caption("ChaseEscapeStrategyEnv")
            self._pygame_clock = pygame.time.Clock()
        screen = self._pygame_screen
        screen.fill((255, 255, 255))
        for wall in self.state.walls:
            if len(wall.points) >= 2:
                pygame.draw.lines(screen, (0, 0, 0), False, wall.points, wall.thickness)
        police = self.state.police_agents[0]
        enemy = self.state.enemy_agents[0]
        pygame.draw.circle(screen, (0, 0, 255), (int(police.position[0]), int(police.position[1])), int(police.radius))
        pygame.draw.circle(screen, (255, 0, 0), (int(enemy.position[0]), int(enemy.position[1])), int(enemy.radius))
        # Visualize current strategy plan (approximate target point)
        if self.last_strategy_action is not None:
            w, h = cfg.width, cfg.height
            left, right, top, bottom = distance_to_rect_edges(enemy.position, w, h)
            if left <= min(right, top, bottom):
                exit_pt: Vec2 = (0.0, enemy.position[1])
            elif right <= min(left, top, bottom):
                exit_pt = (float(w), enemy.position[1])
            elif top <= min(left, right, bottom):
                exit_pt = (enemy.position[0], 0.0)
            else:
                exit_pt = (enemy.position[0], float(h))
            bottleneck = ((enemy.position[0] + exit_pt[0]) / 2, (enemy.position[1] + exit_pt[1]) / 2)
            target: Optional[Vec2]
            color = (0, 0, 0)
            if self.last_strategy_action == 0:  # chase
                target = enemy.position
                color = (0, 120, 255)
            elif self.last_strategy_action == 1:  # block exit
                target = exit_pt
                color = (0, 200, 0)
            elif self.last_strategy_action == 2:  # bottleneck
                target = bottleneck
                color = (255, 140, 0)
            elif self.last_strategy_action == 3:  # hold
                target = None
            elif self.last_strategy_action in (4, 5):  # flanks
                to_p = sub(police.position, enemy.position)
                if self.last_strategy_action == 4:
                    perp = (-to_p[1], to_p[0])
                else:
                    perp = (to_p[1], -to_p[0])
                if length(perp) == 0:
                    target = None
                else:
                    perp_n = normalize(perp)
                    target = add(enemy.position, mul(perp_n, 80))
                color = (160, 32, 240)
            else:
                target = None
            if target is not None:
                pygame.draw.circle(screen, color, (int(target[0]), int(target[1])), 6)
                pygame.draw.line(screen, color, police.position, target, 2)
        pygame.display.flip()
        if self._pygame_clock:
            self._pygame_clock.tick(60)

    def close(self) -> None:
        if self._pygame_screen is not None:
            try:
                import pygame
                pygame.quit()
            except ImportError:
                pass
            self._pygame_screen = None


# -------------------------------------------------------------------------
# Dual env: run both trained models (step 3 – full run, no training)
# -------------------------------------------------------------------------

class ChaseEscapeDualEnv:
    """
    Run both police and enemy with trained models. No training – just step with (police_action, enemy_action).
    reset() -> (police_obs, enemy_obs). step(police_action, enemy_action) -> (police_obs, enemy_obs), done, info.
    """

    def __init__(
        self,
        config: WorldConfig | None = None,
        max_steps: int = 600,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.config = config or DEFAULT_WORLD_CONFIG
        self.max_steps = max_steps
        self._rng = random.Random(seed)
        self.options = options
        self.state: Optional[WorldState] = None
        self.steps: int = 0

    def _build_state(self) -> None:
        cfg = self.config
        opts = self.options or {}
        margin = 100.0
        w, h = cfg.width, cfg.height
        self.state = WorldState(width=w, height=h)
        if opts.get("walls") is not None:
            for wb in opts["walls"]:
                if isinstance(wb, WallStroke):
                    self.state.walls.append(wb)
                else:
                    pts = [tuple(p) for p in wb.get("points", [])]
                    self.state.walls.append(WallStroke(points=pts, thickness=int(wb.get("thickness", 6))))
        else:
            _tmp = ChaseEscapeEnv(self.config, self.max_steps, None, None)
            _tmp._rng = self._rng
            _tmp._generate_random_walls(self.state)
        if opts.get("police_pos") is not None and opts.get("enemy_pos") is not None:
            px, py = opts["police_pos"]
            ex, ey = opts["enemy_pos"]
            police_dir = opts.get("police_dir", 0.0)
            enemy_dir = opts.get("enemy_dir", math.pi)
        else:
            px = self._rng.uniform(margin, w - margin)
            py = self._rng.uniform(margin, h - margin)
            ex = self._rng.uniform(margin, w - margin)
            ey = self._rng.uniform(margin, h - margin)
            police_dir = self._rng.uniform(-math.pi, math.pi)
            enemy_dir = self._rng.uniform(-math.pi, math.pi)
        self.state.police_agents.append(PoliceAgent(
            position=(float(px), float(py)), direction=float(police_dir),
            speed=cfg.police_speed, radius=cfg.police_radius,
            fov_angle=cfg.police_fov_deg, vision_range=cfg.police_vision_range,
            arrest_radius=cfg.police_arrest_radius,
        ))
        self.state.enemy_agents.append(EnemyAgent(
            position=(float(ex), float(ey)), direction=float(enemy_dir),
            speed=cfg.enemy_speed, radius=cfg.enemy_radius,
        ))
        compute_enemy_visibility_and_danger(self.state, self.config)

    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        self._build_state()
        self.steps = 0
        return (
            get_police_obs(self.state, self.config, self.steps, self.max_steps),
            get_enemy_obs(self.state, self.config, self.steps, self.max_steps),
        )

    def _decode_police(self, action: int) -> Tuple[float, float]:
        cfg = self.config
        f, t = 0.0, 0.0
        if action == 1: f = cfg.police_speed
        elif action == 2: t = -0.1
        elif action == 3: t = 0.1
        elif action == 4: f, t = cfg.police_speed, -0.1
        elif action == 5: f, t = cfg.police_speed, 0.1
        return f, t

    def _decode_enemy(self, action: int) -> float:
        enemy = self.state.enemy_agents[0]
        turn = {0: 0.0, 1: 0.0, 2: -0.12, 3: 0.12, 4: -0.12, 5: 0.12}.get(action, 0.0)
        return enemy.direction + turn

    def step(self, police_action: int, enemy_action: int) -> Tuple[np.ndarray, np.ndarray, bool, bool, Dict[str, Any]]:
        assert self.state is not None
        self.steps += 1
        cfg = self.config
        police = self.state.police_agents[0]
        enemy = self.state.enemy_agents[0]
        police_actions = [self._decode_police(police_action)]
        enemy_dirs = [self._decode_enemy(enemy_action)]
        terminated, info = update_world(self.state, police_actions=police_actions, enemy_dirs=enemy_dirs, config=cfg)
        compute_enemy_visibility_and_danger(self.state, cfg)
        truncated = self.steps >= self.max_steps
        po = get_police_obs(self.state, cfg, self.steps, self.max_steps)
        eo = get_enemy_obs(self.state, cfg, self.steps, self.max_steps)
        return po, eo, terminated, truncated, info


# -------------------------------------------------------------------------
# Enemy training env: RL agent = enemy (goal: escape), police = scripted or trained model
# -------------------------------------------------------------------------

class ChaseEscapeEnemyEnv(gym.Env):
    """
    Enemy is the RL agent (goal: escape off the map).
    Police: scripted (chase) if police_model is None; else use trained police_model (step 2).
    Train enemy and save as ppo_enemy_escape_final.zip.
    """

    metadata = {"render_modes": ["human", "none"], "render_fps": 60}

    def __init__(
        self,
        config: WorldConfig | None = None,
        max_steps: int = 600,
        render_mode: str | None = None,
        seed: Optional[int] = None,
        police_model: Optional[Any] = None,
    ) -> None:
        super().__init__()
        self.config = config or DEFAULT_WORLD_CONFIG
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.police_model = police_model  # If set, use trained police; else scripted chase
        self.action_space = spaces.Discrete(6)
        # Enemy obs = [ex, ey, cos_dir, sin_dir, rel_px, rel_py, dist_p, dist_exit, 4 wall_rays, t_rem] -> 13 dims
        obs_dim = 13
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        self._rng = random.Random(seed)
        self.state: Optional[WorldState] = None
        self.steps: int = 0
        self._pygame_screen = None
        self._pygame_clock = None

    def seed(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)

    def _generate_random_walls(self, state: WorldState) -> None:
        cfg = self.config
        w, h = state.width, state.height
        margin = 60.0
        num_strokes = self._rng.randint(4, 9)
        stroke_length = 220.0
        for _ in range(num_strokes):
            x0 = self._rng.uniform(margin, w - margin)
            y0 = self._rng.uniform(margin, h - margin)
            length = self._rng.uniform(stroke_length * 0.5, stroke_length * 1.2)
            if self._rng.random() < 0.35:
                points = [(x0, y0)]
                for _ in range(2):
                    angle = self._rng.uniform(-math.pi, math.pi)
                    seg_len = length * self._rng.uniform(0.3, 0.7)
                    x1 = max(0.0, min(w, points[-1][0] + math.cos(angle) * seg_len))
                    y1 = max(0.0, min(h, points[-1][1] + math.sin(angle) * seg_len))
                    points.append((x1, y1))
                state.walls.append(WallStroke(points=points, thickness=6))
            else:
                angle = self._rng.uniform(-math.pi, math.pi)
                x1 = max(0.0, min(w, x0 + math.cos(angle) * length))
                y1 = max(0.0, min(h, y0 + math.sin(angle) * length))
                state.walls.append(WallStroke(points=[(x0, y0), (x1, y1)], thickness=6))

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is not None:
            self.seed(seed)
        if options is None and getattr(self, "training_map_options", None) is not None:
            options = self.training_map_options
        cfg = self.config
        self.state = WorldState(width=cfg.width, height=cfg.height)
        self.state.police_agents.clear()
        self.state.enemy_agents.clear()
        self.state.walls.clear()
        self.steps = 0
        opts = options or {}
        margin = 100.0
        w, h = cfg.width, cfg.height
        if opts.get("walls") is not None:
            for wb in opts["walls"]:
                if isinstance(wb, WallStroke):
                    self.state.walls.append(wb)
                else:
                    pts = [tuple(p) for p in wb.get("points", [])]
                    self.state.walls.append(WallStroke(points=pts, thickness=int(wb.get("thickness", 6))))
        else:
            self._generate_random_walls(self.state)
        if opts.get("police_pos") is not None and opts.get("enemy_pos") is not None:
            px, py = opts["police_pos"]
            ex, ey = opts["enemy_pos"]
            police_dir = opts.get("police_dir", 0.0)
            enemy_dir = opts.get("enemy_dir", math.pi)
        else:
            px = self._rng.uniform(margin, w - margin)
            py = self._rng.uniform(margin, h - margin)
            ex = self._rng.uniform(margin, w - margin)
            ey = self._rng.uniform(margin, h - margin)
            police_dir = self._rng.uniform(-math.pi, math.pi)
            enemy_dir = self._rng.uniform(-math.pi, math.pi)
        self.state.police_agents.append(PoliceAgent(
            position=(float(px), float(py)), direction=float(police_dir),
            speed=cfg.police_speed, radius=cfg.police_radius,
            fov_angle=cfg.police_fov_deg, vision_range=cfg.police_vision_range,
            arrest_radius=cfg.police_arrest_radius,
        ))
        self.state.enemy_agents.append(EnemyAgent(
            position=(float(ex), float(ey)), direction=float(enemy_dir),
            speed=cfg.enemy_speed, radius=cfg.enemy_radius,
        ))
        compute_enemy_visibility_and_danger(self.state, self.config)
        return self._get_obs(), {}

    def _decode_enemy_action(self, action: int) -> float:
        """Map discrete action to angular delta for enemy direction."""
        turn = {0: 0.0, 1: 0.0, 2: -0.12, 3: 0.12, 4: -0.12, 5: 0.12}.get(action, 0.0)
        enemy = self.state.enemy_agents[0]
        return enemy.direction + turn

    def _decode_police_action(self, action: int) -> Tuple[float, float]:
        """Map discrete police action to (forward_speed, angular_delta)."""
        cfg = self.config
        forward = 0.0
        turn = 0.0
        if action == 1:
            forward = cfg.police_speed
        elif action == 2:
            turn = -0.1
        elif action == 3:
            turn = 0.1
        elif action == 4:
            forward = cfg.police_speed
            turn = -0.1
        elif action == 5:
            forward = cfg.police_speed
            turn = 0.1
        return forward, turn

    def step(self, action: int):
        assert self.state is not None
        cfg = self.config
        self.steps += 1
        police = self.state.police_agents[0]
        enemy = self.state.enemy_agents[0]
        if self.police_model is not None:
            police_obs = get_police_obs(self.state, cfg, self.steps - 1, self.max_steps)
            police_action, _ = self.police_model.predict(police_obs, deterministic=True)
            police_actions = [self._decode_police_action(int(police_action))]
        else:
            police_actions = [scripted_police_chase(police, enemy)]
        enemy_dirs = [self._decode_enemy_action(action)]
        terminated, info = update_world(
            self.state, police_actions=police_actions, enemy_dirs=enemy_dirs, config=cfg,
        )
        compute_enemy_visibility_and_danger(self.state, cfg)
        arrested = info.get("arrested", False)
        escaped = info.get("escaped", False)
        reward = 0.0
        if escaped:
            reward += 5.0
        if arrested:
            reward -= 5.0
        # Negative reward when enemy movement was blocked by walls (stuck).
        if enemy.blocked_last_step and not (escaped or arrested):
            reward -= 0.05
        reward -= 0.002
        truncated = self.steps >= self.max_steps
        if truncated and not (arrested or escaped):
            reward -= 0.5
        return self._get_obs(), reward, terminated, truncated, {"arrested": arrested, "escaped": escaped}

    def _get_obs(self) -> np.ndarray:
        assert self.state is not None
        cfg = self.config
        police = self.state.police_agents[0]
        enemy = self.state.enemy_agents[0]
        w, h = cfg.width, cfg.height
        ex = enemy.position[0] / w
        ey = enemy.position[1] / h
        dc = math.cos(enemy.direction)
        ds = math.sin(enemy.direction)
        rel_px = (police.position[0] - enemy.position[0]) / w
        rel_py = (police.position[1] - enemy.position[1]) / h
        dist_p = distance(police.position, enemy.position) / math.hypot(w, h)
        dist_exit = nearest_exit_distance(enemy.position, w, h) / max(w, h)
        left, right, top, bottom = distance_to_rect_edges(enemy.position, w, h)
        wall_rays = [left / w, right / w, top / h, bottom / h]
        t_rem = 1.0 - (self.steps / max(1, self.max_steps))
        return np.array(
            [ex, ey, dc, ds, rel_px, rel_py, dist_p, dist_exit, *wall_rays, t_rem],
            dtype=np.float32,
        )

    def render(self) -> None:
        if self.render_mode != "human" or self.state is None:
            return
        try:
            import pygame
        except ImportError:
            return
        cfg = self.config
        if self._pygame_screen is None:
            pygame.init()
            self._pygame_screen = pygame.display.set_mode((cfg.width, cfg.height))
            pygame.display.set_caption("ChaseEscapeEnemyEnv")
            self._pygame_clock = pygame.time.Clock()
        screen = self._pygame_screen
        screen.fill((255, 255, 255))
        # Escape zone border on all 4 edges
        escape_band = 12
        escape_surf = pygame.Surface((cfg.width, cfg.height), pygame.SRCALPHA)
        escape_surf.fill((100, 255, 100, 90), (0, 0, cfg.width, escape_band))
        escape_surf.fill((100, 255, 100, 90), (0, cfg.height - escape_band, cfg.width, escape_band))
        escape_surf.fill((100, 255, 100, 90), (0, 0, escape_band, cfg.height))
        escape_surf.fill((100, 255, 100, 90), (cfg.width - escape_band, 0, escape_band, cfg.height))
        screen.blit(escape_surf, (0, 0))
        for wall in self.state.walls:
            if len(wall.points) >= 2:
                pygame.draw.lines(screen, (0, 0, 0), False, wall.points, wall.thickness)
        police = self.state.police_agents[0]
        enemy = self.state.enemy_agents[0]
        pygame.draw.circle(screen, (0, 0, 255), (int(police.position[0]), int(police.position[1])), int(police.radius))
        pygame.draw.circle(screen, (255, 0, 0), (int(enemy.position[0]), int(enemy.position[1])), int(enemy.radius))
        pygame.display.flip()
        if self._pygame_clock:
            self._pygame_clock.tick(60)

    def close(self) -> None:
        if self._pygame_screen is not None:
            try:
                import pygame
                pygame.quit()
            except ImportError:
                pass
            self._pygame_screen = None


def make_env(render_mode: str | None = None, seed: Optional[int] = None) -> ChaseEscapeEnv:
    """Helper for Stable-Baselines3."""
    return ChaseEscapeEnv(render_mode=render_mode, seed=seed)


def load_training_map(path: str | Path) -> Dict[str, Any]:
    """
    Load a saved map JSON (from play_model "Save map" or sandbox) for training.

    Returns an options dict you can pass to env.reset(options=...).
    JSON format: width, height, police: [{x, y, direction}], enemies: [{x, y, direction}],
    walls: [{points: [[x,y],...], thickness}].
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Training map not found: {p}")
    data = json.loads(p.read_text())
    w = data.get("width", 1280)
    h = data.get("height", 720)
    police_list = data.get("police", [])
    enemy_list = data.get("enemies", [])
    if not police_list or not enemy_list:
        raise ValueError("Map must have at least one police and one enemy.")
    p0 = police_list[0]
    e0 = enemy_list[0]
    # Support both {"x","y"} and {"x","y","direction"}
    police_pos = (float(p0["x"]), float(p0["y"]))
    police_dir = float(p0.get("direction", 0.0))
    enemy_pos = (float(e0["x"]), float(e0["y"]))
    enemy_dir = float(e0.get("direction", math.pi))
    walls = []
    for wb in data.get("walls", []):
        pts = wb.get("points", [])
        if isinstance(pts[0] if pts else None, dict):
            points = [(float(pt["x"]), float(pt["y"])) for pt in pts]
        else:
            points = [(float(pt[0]), float(pt[1])) for pt in pts]
        walls.append({"points": points, "thickness": int(wb.get("thickness", 6))})
    return {
        "walls": walls,
        "police_pos": police_pos,
        "police_dir": police_dir,
        "enemy_pos": enemy_pos,
        "enemy_dir": enemy_dir,
    }

