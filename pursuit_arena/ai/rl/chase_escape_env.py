from __future__ import annotations

"""
Gymnasium-compatible environment for a single police agent chasing a scripted enemy.
"""

from typing import Any, Dict, Optional, Tuple
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
    angle_to_vector,
    clamp_to_bounds,
    distance,
    nearest_exit_distance,
    distance_to_rect_edges,
)
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

        cfg = self.config
        self.state = WorldState(width=cfg.width, height=cfg.height)
        self.state.police_agents.clear()
        self.state.enemy_agents.clear()
        self.state.walls.clear()
        self.steps = 0

        # Random training walls
        self._generate_random_walls(self.state)

        # Random spawn positions away from edges
        margin = 100.0
        px = self._rng.uniform(margin, cfg.width - margin)
        py = self._rng.uniform(margin, cfg.height - margin)
        ex = self._rng.uniform(margin, cfg.width - margin)
        ey = self._rng.uniform(margin, cfg.height - margin)

        police = PoliceAgent(
            position=(px, py),
            direction=self._rng.uniform(-math.pi, math.pi),
            speed=cfg.police_speed,
            radius=cfg.police_radius,
            fov_angle=cfg.police_fov_deg,
            vision_range=cfg.police_vision_range,
            arrest_radius=cfg.police_arrest_radius,
        )
        enemy = EnemyAgent(
            position=(ex, ey),
            direction=self._rng.uniform(-math.pi, math.pi),
            speed=cfg.enemy_speed,
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

        # Dense shaping
        enemy_visible = police_can_see_enemy(police, enemy, self.state.walls)
        if enemy_visible:
            # Encourage keeping enemy visible and approaching
            reward += 0.01

        # Distance-based shaping
        d_prev = distance(police.position, enemy.position)
        # simple projection for previous step position (approximate)
        # here we just use current distance; for proper delta you'd track previous.
        d_now = d_prev
        if enemy_visible:
            reward += 0.001 * (1.0 / max(d_now, 1.0))

        # Small negative step cost
        reward -= 0.001

        if arrested:
            reward += 10.0
        if escaped:
            reward -= 10.0

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


def make_env(render_mode: str | None = None, seed: Optional[int] = None) -> ChaseEscapeEnv:
    """Helper for Stable-Baselines3."""
    return ChaseEscapeEnv(render_mode=render_mode, seed=seed)

