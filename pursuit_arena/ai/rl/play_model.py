from __future__ import annotations

"""
Load a trained model and run it with a UI: Start / Stop / Restart and Edit mode
to draw walls and place police and enemy before playing.
Save map writes a JSON you can use for training (same layout every episode).
"""

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pygame
from stable_baselines3 import PPO

from ...core.config import DEFAULT_WORLD_CONFIG
from ...core.entities import WallStroke
from .chase_escape_env import ChaseEscapeEnv


Vec2 = Tuple[float, float]

# Default path for map used in play_model and for training
TRAINING_MAP_PATH = Path("training_map.json")

# UI
WIN_W = 1280
WIN_H = 720
BAR_H = 40
BUTTONS_Y = WIN_H  # buttons just below game area
BTN_H = 32
BTN_W = 90
BTN_MARGIN = 10


def _build_options(
    walls: List[WallStroke],
    police_pos: Vec2 | None,
    police_dir: float,
    enemy_pos: Vec2 | None,
    enemy_dir: float,
) -> Dict[str, Any] | None:
    if police_pos is None or enemy_pos is None:
        return None
    return {
        "walls": [{"points": list(w.points), "thickness": w.thickness} for w in walls],
        "police_pos": police_pos,
        "police_dir": police_dir,
        "enemy_pos": enemy_pos,
        "enemy_dir": enemy_dir,
    }


def _save_training_map(
    path: Path,
    walls: List[WallStroke],
    police_pos: Vec2 | None,
    police_dir: float,
    enemy_pos: Vec2 | None,
    enemy_dir: float,
) -> bool:
    """Save current layout to JSON for training. Returns True if saved."""
    if police_pos is None or enemy_pos is None:
        return False
    data = {
        "width": WIN_W,
        "height": WIN_H,
        "police": [{"x": police_pos[0], "y": police_pos[1], "direction": police_dir}],
        "enemies": [{"x": enemy_pos[0], "y": enemy_pos[1], "direction": enemy_dir}],
        "walls": [
            {"points": [[round(x, 1), round(y, 1)] for (x, y) in w.points], "thickness": w.thickness}
            for w in walls
        ],
    }
    path.write_text(json.dumps(data, indent=2))
    return True


def _load_training_map(path: Path) -> tuple[List[WallStroke], Vec2 | None, float, Vec2 | None, float]:
    """Load layout from JSON. Returns (walls, police_pos, police_dir, enemy_pos, enemy_dir)."""
    if not path.exists():
        return [], None, 0.0, None, math.pi
    data = json.loads(path.read_text())
    walls = []
    for wb in data.get("walls", []):
        pts = wb.get("points", [])
        if pts and isinstance(pts[0], dict):
            points = [(float(p["x"]), float(p["y"])) for p in pts]
        else:
            points = [(float(p[0]), float(p[1])) for p in pts]
        walls.append(WallStroke(points=points, thickness=int(wb.get("thickness", 6))))
    police_pos = police_dir = enemy_pos = enemy_dir = None
    if data.get("police"):
        p0 = data["police"][0]
        police_pos = (float(p0["x"]), float(p0["y"]))
        police_dir = float(p0.get("direction", 0.0))
    if data.get("enemies"):
        e0 = data["enemies"][0]
        enemy_pos = (float(e0["x"]), float(e0["y"]))
        enemy_dir = float(e0.get("direction", math.pi))
    return (
        walls,
        police_pos,
        police_dir if police_dir is not None else 0.0,
        enemy_pos,
        enemy_dir if enemy_dir is not None else math.pi,
    )


def main() -> None:
    model_path = Path("runs/ppo_chase_escape/ppo_chase_escape_final.zip")
    if not model_path.exists():
        raise SystemExit(f"Model file not found at {model_path}. Train first with train_ppo.py.")

    env = ChaseEscapeEnv(render_mode=None)
    model = PPO.load(str(model_path), env=env)

    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H + BAR_H))
    pygame.display.set_caption("Play Model — Edit / Start / Stop / Restart")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 14)

    # Edit state
    walls: List[WallStroke] = []
    current_stroke: List[Vec2] = []
    police_pos: Vec2 | None = None
    police_dir: float = 0.0
    enemy_pos: Vec2 | None = None
    enemy_dir: float = math.pi

    # Run state
    mode = "edit"  # "edit" | "running" | "stopped"
    saved_layout: Dict[str, Any] | None = None
    obs = None
    done = False
    truncated = False

    def draw_game_area(state_source: str) -> None:
        """Draw walls and agents; state_source is 'env' or 'edit'."""
        # Clear game area
        screen.fill((255, 255, 255), (0, 0, WIN_W, WIN_H))

        if state_source == "edit":
            for w in walls:
                if len(w.points) >= 2:
                    pygame.draw.lines(screen, (0, 0, 0), False, w.points, w.thickness)
            if len(current_stroke) >= 2:
                pygame.draw.lines(screen, (80, 80, 80), False, current_stroke, 4)
            if police_pos is not None:
                pygame.draw.circle(screen, (0, 0, 255), (int(police_pos[0]), int(police_pos[1])), 10)
                end = (
                    police_pos[0] + math.cos(police_dir) * 25,
                    police_pos[1] + math.sin(police_dir) * 25,
                )
                pygame.draw.line(screen, (0, 0, 180), police_pos, end, 2)
            if enemy_pos is not None:
                pygame.draw.circle(screen, (255, 0, 0), (int(enemy_pos[0]), int(enemy_pos[1])), 10)
        else:
            # From env.state
            s = env.state
            if s is None:
                return
            for w in s.walls:
                if len(w.points) >= 2:
                    pygame.draw.lines(screen, (0, 0, 0), False, w.points, w.thickness)
            if s.police_agents:
                p = s.police_agents[0]
                # FOV
                half_fov = math.radians(p.fov_angle / 2.0)
                p0 = p.position
                p1 = (p0[0] + math.cos(p.direction - half_fov) * p.vision_range,
                      p0[1] + math.sin(p.direction - half_fov) * p.vision_range)
                p2 = (p0[0] + math.cos(p.direction + half_fov) * p.vision_range,
                      p0[1] + math.sin(p.direction + half_fov) * p.vision_range)
                fov_surf = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
                pygame.draw.polygon(fov_surf, (255, 255, 0, 80), [p0, p1, p2])
                screen.blit(fov_surf, (0, 0))
                pygame.draw.circle(screen, (0, 0, 255), (int(p.position[0]), int(p.position[1])), int(p.radius))
                end = (p.position[0] + math.cos(p.direction) * 25, p.position[1] + math.sin(p.direction) * 25)
                pygame.draw.line(screen, (0, 0, 180), p.position, end, 2)
            if s.enemy_agents:
                e = s.enemy_agents[0]
                pygame.draw.circle(screen, (255, 0, 0), (int(e.position[0]), int(e.position[1])), int(e.radius))

    # Button rects (fixed positions)
    by = WIN_H + (BAR_H - BTN_H) // 2
    button_edit_rect = pygame.Rect(BTN_MARGIN, by, BTN_W, BTN_H)
    button_start_rect = pygame.Rect(BTN_MARGIN + BTN_W + BTN_MARGIN, by, BTN_W, BTN_H)
    button_stop_rect = pygame.Rect(BTN_MARGIN + 2 * (BTN_W + BTN_MARGIN), by, BTN_W, BTN_H)
    button_restart_rect = pygame.Rect(BTN_MARGIN + 3 * (BTN_W + BTN_MARGIN), by, BTN_W, BTN_H)
    button_save_rect = pygame.Rect(BTN_MARGIN + 4 * (BTN_W + BTN_MARGIN), by, 88, BTN_H)   # "Save map"
    button_load_rect = pygame.Rect(BTN_MARGIN + 4 * (BTN_W + BTN_MARGIN) + 88 + BTN_MARGIN, by, 88, BTN_H)

    def draw_bar() -> None:
        screen.fill((240, 240, 240), (0, WIN_H, WIN_W, BAR_H))
        for label, r in [
            ("Edit", button_edit_rect),
            ("Start", button_start_rect),
            ("Stop", button_stop_rect),
            ("Restart", button_restart_rect),
            ("Save map", button_save_rect),
            ("Load map", button_load_rect),
        ]:
            color = (180, 180, 255) if (label.lower() == mode) else (220, 220, 220)
            pygame.draw.rect(screen, color, r)
            pygame.draw.rect(screen, (0, 0, 0), r, 1)
            t = font.render(label, True, (0, 0, 0))
            screen.blit(t, (r.x + (r.w - t.get_width()) // 2, r.y + (r.h - t.get_height()) // 2))
        x = button_load_rect.right + BTN_MARGIN * 2
        if mode == "edit":
            hint = "L-click: police  R-click: enemy  Ctrl+drag: wall"
        elif mode == "running":
            hint = "Running — click Stop to pause"
        else:
            hint = "Stopped — click Restart or Edit to change layout"
        h = font.render(hint, True, (60, 60, 60))
        screen.blit(h, (x, WIN_H + (BAR_H - h.get_height()) // 2))

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
                break

            if event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                if my >= WIN_H:
                    # Button row
                    if button_edit_rect.collidepoint(mx, my):
                        mode = "edit"
                        done = False
                        truncated = False
                    elif button_start_rect.collidepoint(mx, my):
                        opts = _build_options(walls, police_pos, police_dir, enemy_pos, enemy_dir)
                        if opts is None:
                            pass  # need both agents; could show message
                        else:
                            saved_layout = opts
                            obs, _ = env.reset(options=opts)
                            mode = "running"
                            done = False
                            truncated = False
                    elif button_stop_rect.collidepoint(mx, my):
                        mode = "stopped"
                    elif button_restart_rect.collidepoint(mx, my):
                        if saved_layout is not None:
                            obs, _ = env.reset(options=saved_layout)
                            mode = "running"
                            done = False
                            truncated = False
                    elif button_save_rect.collidepoint(mx, my):
                        if _save_training_map(TRAINING_MAP_PATH, walls, police_pos, police_dir, enemy_pos, enemy_dir):
                            saved_layout = _build_options(walls, police_pos, police_dir, enemy_pos, enemy_dir)
                            print(f"Saved map to {TRAINING_MAP_PATH} (use for training)")
                        else:
                            print("Place police and enemy first, then Save map")
                    elif button_load_rect.collidepoint(mx, my):
                        walls, police_pos, police_dir, enemy_pos, enemy_dir = _load_training_map(TRAINING_MAP_PATH)
                        if police_pos is not None or enemy_pos is not None or walls:
                            current_stroke = []
                            if police_pos is not None:
                                saved_layout = _build_options(walls, police_pos, police_dir, enemy_pos, enemy_dir)
                            print(f"Loaded map from {TRAINING_MAP_PATH}")
                        else:
                            print(f"No map at {TRAINING_MAP_PATH}")
                elif mode == "edit" and my < WIN_H:
                    mods = pygame.key.get_mods()
                    if event.button == 1:  # left
                        if mods & pygame.KMOD_CTRL:
                            current_stroke = [(mx, my)]
                        else:
                            police_pos = (float(mx), float(my))
                    elif event.button == 3:  # right
                        enemy_pos = (float(mx), float(my))

            if event.type == pygame.MOUSEMOTION and mode == "edit" and pygame.mouse.get_pressed()[0]:
                if pygame.key.get_mods() & pygame.KMOD_CTRL and current_stroke:
                    current_stroke.append(event.pos)

            if event.type == pygame.MOUSEBUTTONUP and mode == "edit" and event.button == 1 and current_stroke:
                if len(current_stroke) >= 2:
                    walls.append(WallStroke(points=list(current_stroke)))
                current_stroke = []

        # Step simulation when running
        if mode == "running" and obs is not None:
            if not (done or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, _, done, truncated, _ = env.step(int(action))
            if done or truncated:
                mode = "stopped"

        if mode == "edit":
            draw_game_area("edit")
        else:
            draw_game_area("env")
        draw_bar()

        pygame.display.flip()
        clock.tick(60)

    env.close()
    pygame.quit()


if __name__ == "__main__":
    main()
