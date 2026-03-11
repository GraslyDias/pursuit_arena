from __future__ import annotations

"""
Pygame-based 2D sandbox editor and simulator.

Controls
--------
- Left click: add police agent
- Right click: add enemy agent
- Ctrl + left drag: draw walls/freehand obstacles
- R: reset simulation state (keep map)
- C: clear all (agents and walls)
- S: save map to JSON
- L: load map from JSON
- SPACE: start/stop simulation

This module is intentionally simple and beginner-friendly.
"""

from dataclasses import asdict
from pathlib import Path
from typing import List, Tuple
import json
import math

import pygame

from ..core.config import DEFAULT_WORLD_CONFIG
from ..core.entities import EnemyAgent, PoliceAgent, WallStroke, WorldState
from ..core.enemy_ai import choose_enemy_directions
from ..core.world import update_world, compute_enemy_visibility_and_danger


Vec2 = Tuple[float, float]


class SandboxApp:
    """Main Pygame application for the sandbox editor."""

    def __init__(self, width: int = DEFAULT_WORLD_CONFIG.width, height: int = DEFAULT_WORLD_CONFIG.height) -> None:
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Pursuit Arena - Sandbox Editor")
        self.clock = pygame.time.Clock()

        self.font = pygame.font.SysFont("consolas", 16)

        self.state = WorldState(width=self.width, height=self.height)
        self.simulation_running = False

        # Manual control: index of the controlled police agent (if any)
        self.controlled_police_index: int | None = 0

        self.current_wall_points: List[Vec2] = []

    # ------------------------------------------------------------------
    # Map persistence
    # ------------------------------------------------------------------
    def save_map(self, path: Path) -> None:
        data = {
            "width": self.state.width,
            "height": self.state.height,
            "police": [
                {
                    "x": p.position[0],
                    "y": p.position[1],
                }
                for p in self.state.police_agents
            ],
            "enemies": [
                {
                    "x": e.position[0],
                    "y": e.position[1],
                }
                for e in self.state.enemy_agents
            ],
            "walls": [
                {"points": [{"x": x, "y": y} for (x, y) in w.points], "thickness": w.thickness}
                for w in self.state.walls
            ],
        }
        path.write_text(json.dumps(data, indent=2))

    def load_map(self, path: Path) -> None:
        if not path.exists():
            print(f"No map file at {path}")
            return
        data = json.loads(path.read_text())
        self.state.width = data.get("width", self.width)
        self.state.height = data.get("height", self.height)
        self.state.police_agents.clear()
        self.state.enemy_agents.clear()
        self.state.walls.clear()

        for p in data.get("police", []):
            self.add_police((float(p["x"]), float(p["y"])))
        for e in data.get("enemies", []):
            self.add_enemy((float(e["x"]), float(e["y"])))
        for w in data.get("walls", []):
            points = [(float(pt["x"]), float(pt["y"])) for pt in w.get("points", [])]
            self.state.walls.append(WallStroke(points=points, thickness=int(w.get("thickness", 6))))

    # ------------------------------------------------------------------
    # Entity creation helpers
    # ------------------------------------------------------------------
    def add_police(self, pos: Vec2) -> None:
        cfg = DEFAULT_WORLD_CONFIG
        self.state.police_agents.append(
            PoliceAgent(
                position=pos,
                direction=0.0,
                speed=cfg.police_speed,
                radius=cfg.police_radius,
                fov_angle=cfg.police_fov_deg,
                vision_range=cfg.police_vision_range,
                arrest_radius=cfg.police_arrest_radius,
            )
        )

    def add_enemy(self, pos: Vec2) -> None:
        cfg = DEFAULT_WORLD_CONFIG
        self.state.enemy_agents.append(
            EnemyAgent(
                position=pos,
                direction=math.pi,  # face left by default
                speed=cfg.enemy_speed,
                radius=cfg.enemy_radius,
            )
        )

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------
    def handle_events(self) -> bool:
        """Handle Pygame events. Return False to quit."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_SPACE:
                    self.simulation_running = not self.simulation_running
                elif event.key == pygame.K_r:
                    # Reset simulation but keep map
                    self.state.step_count = 0
                    self.simulation_running = False
                elif event.key == pygame.K_TAB:
                    # Cycle which police agent is controlled manually
                    if self.state.police_agents:
                        if self.controlled_police_index is None:
                            self.controlled_police_index = 0
                        else:
                            self.controlled_police_index = (self.controlled_police_index + 1) % len(
                                self.state.police_agents
                            )
                elif event.key == pygame.K_c:
                    # Clear everything
                    self.state = WorldState(width=self.width, height=self.height)
                    self.simulation_running = False
                elif event.key == pygame.K_s:
                    self.save_map(Path("sandbox_map.json"))
                    print("Saved map to sandbox_map.json")
                elif event.key == pygame.K_l:
                    self.load_map(Path("sandbox_map.json"))
                    print("Loaded map from sandbox_map.json")

            # Mouse handling
            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                mods = pygame.key.get_mods()
                if event.button == 1:  # left
                    if mods & pygame.KMOD_CTRL:
                        # start a new wall stroke
                        self.current_wall_points = [pos]
                    else:
                        self.add_police(pos)
                elif event.button == 3:  # right
                    self.add_enemy(pos)

            if event.type == pygame.MOUSEMOTION:
                if pygame.mouse.get_pressed()[0] and (pygame.key.get_mods() & pygame.KMOD_CTRL):
                    pos = pygame.mouse.get_pos()
                    self.current_wall_points.append(pos)

            if event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1 and self.current_wall_points:
                    # finish wall stroke
                    if len(self.current_wall_points) >= 2:
                        self.state.walls.append(WallStroke(points=list(self.current_wall_points)))
                    self.current_wall_points = []

        return True

    # ------------------------------------------------------------------
    # Simulation update
    # ------------------------------------------------------------------
    def step_simulation(self) -> None:
        # Manual control for one police agent using WASD / arrow keys
        police_actions: List[Tuple[float, float]] = [(0.0, 0.0) for _ in self.state.police_agents]

        keys = pygame.key.get_pressed()
        if self.controlled_police_index is not None and 0 <= self.controlled_police_index < len(
            self.state.police_agents
        ):
            idx = self.controlled_police_index
            forward = 0.0
            turn = 0.0
            if keys[pygame.K_w] or keys[pygame.K_UP]:
                forward += DEFAULT_WORLD_CONFIG.police_speed
            if keys[pygame.K_s] or keys[pygame.K_DOWN]:
                forward -= DEFAULT_WORLD_CONFIG.police_speed
            if keys[pygame.K_a] or keys[pygame.K_LEFT]:
                turn -= 0.08
            if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
                turn += 0.08
            police_actions[idx] = (forward, turn)

        if not self.simulation_running:
            # Apply manual movement but do not move enemies when paused
            terminated, info = update_world(self.state, police_actions=police_actions, enemy_dirs=None)
            compute_enemy_visibility_and_danger(self.state)
            return

        # Scripted enemies when simulation is running
        enemy_dirs = choose_enemy_directions(self.state)

        terminated, info = update_world(self.state, police_actions=police_actions, enemy_dirs=enemy_dirs)
        compute_enemy_visibility_and_danger(self.state)
        if terminated:
            print(f"Terminal event: {info}")
            self.simulation_running = False

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def draw(self) -> None:
        self.screen.fill((255, 255, 255))

        # Escape zone border on all 4 edges (enemy escapes when crossing)
        escape_band = 12
        escape_surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        escape_surf.fill((100, 255, 100, 90), (0, 0, self.width, escape_band))
        escape_surf.fill((100, 255, 100, 90), (0, self.height - escape_band, self.width, escape_band))
        escape_surf.fill((100, 255, 100, 90), (0, 0, escape_band, self.height))
        escape_surf.fill((100, 255, 100, 90), (self.width - escape_band, 0, escape_band, self.height))
        self.screen.blit(escape_surf, (0, 0))

        # Semi-transparent FOV overlay surface
        fov_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)

        # Draw walls
        for wall in self.state.walls:
            if len(wall.points) >= 2:
                pygame.draw.lines(self.screen, (0, 0, 0), False, wall.points, wall.thickness)

        # Current drawing stroke
        if len(self.current_wall_points) >= 2:
            pygame.draw.lines(self.screen, (50, 50, 50), False, self.current_wall_points, 4)

        # Draw police
        for p in self.state.police_agents:
            # FOV triangle (yellow, low opacity)
            half_fov_rad = math.radians(p.fov_angle / 2.0)
            left_dir = p.direction - half_fov_rad
            right_dir = p.direction + half_fov_rad
            p0 = p.position
            p1 = (
                p0[0] + math.cos(left_dir) * p.vision_range,
                p0[1] + math.sin(left_dir) * p.vision_range,
            )
            p2 = (
                p0[0] + math.cos(right_dir) * p.vision_range,
                p0[1] + math.sin(right_dir) * p.vision_range,
            )
            pygame.draw.polygon(fov_surface, (255, 255, 0, 80), [p0, p1, p2])

            pygame.draw.circle(self.screen, (0, 0, 255), (int(p.position[0]), int(p.position[1])), int(p.radius))
            # Draw facing direction
            end = (
                p.position[0] + math.cos(p.direction) * (p.radius * 2),
                p.position[1] + math.sin(p.direction) * (p.radius * 2),
            )
            pygame.draw.line(self.screen, (0, 0, 180), p.position, end, 2)

        # Blit FOV overlay on top
        self.screen.blit(fov_surface, (0, 0))

        # Draw enemies
        for e in self.state.enemy_agents:
            color = (255, 0, 0)
            pygame.draw.circle(self.screen, color, (int(e.position[0]), int(e.position[1])), int(e.radius))

        # HUD
        self.draw_hud()

        pygame.display.flip()

    def draw_hud(self) -> None:
        lines = [
            "Controls:",
            "L-click: add police | R-click: add enemy",
            "Ctrl+L-drag: draw walls",
            "R: reset sim | C: clear all",
            "S: save map | L: load map",
            "SPACE: start/stop simulation | TAB: switch controlled police",
            "",
            f"Police: {len(self.state.police_agents)}  Enemies: {len(self.state.enemy_agents)}",
            f"Walls: {len(self.state.walls)}  Running: {self.simulation_running}",
        ]
        y = 5
        for line in lines:
            surf = self.font.render(line, True, (0, 0, 0))
            self.screen.blit(surf, (5, y))
            y += surf.get_height() + 2

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self) -> None:
        running = True
        while running:
            running = self.handle_events()
            self.step_simulation()
            self.draw()
            self.clock.tick(60)

        pygame.quit()


def main() -> None:
    app = SandboxApp()
    app.run()


if __name__ == "__main__":
    main()

