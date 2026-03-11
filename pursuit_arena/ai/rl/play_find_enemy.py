from __future__ import annotations

"""
Play Find Enemy: place police, place enemy (static dot), draw walls, then run
the trained find-enemy model. Police moves to find and reach the stationary enemy.
UI: Edit mode (L-click police, R-click enemy, Ctrl+drag walls), Start, Stop, Restart,
Save map, Load map, Clear.
"""

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pygame
from stable_baselines3 import PPO

from ...core.entities import WallStroke
from .chase_escape_env import ChaseEscapeEnv, load_training_map


Vec2 = Tuple[float, float]

WIN_W = 1280
WIN_H = 720
BAR_H = 40
BTN_H = 32
BTN_W = 90
BTN_MARGIN = 10

TRAINING_MAP_PATH = Path("training_map.json")


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
    if not path.exists():
        return [], None, 0.0, None, math.pi
    data = json.loads(path.read_text())
    walls: List[WallStroke] = []
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
    model_path = Path("runs/ppo_find_enemy/ppo_find_enemy_final.zip")
    if not model_path.exists():
        raise SystemExit(
            f"Find-enemy model not found at {model_path}. Train first with find_enemy_training.ipynb."
        )

    env = ChaseEscapeEnv(render_mode=None)
    env.static_enemy = True  # enemy is a static dot; only police moves
    model = PPO.load(str(model_path), env=env)

    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H + BAR_H))
    pygame.display.set_caption("Play Find Enemy — Place police & enemy, draw walls, Play / Stop")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 14)

    walls: List[WallStroke] = []
    current_stroke: List[Vec2] = []
    police_pos: Vec2 | None = None
    police_dir: float = 0.0
    enemy_pos: Vec2 | None = None
    enemy_dir: float = math.pi

    mode = "edit"  # "edit" | "running" | "stopped"
    saved_layout: Dict[str, Any] | None = None
    obs = None
    done = False
    truncated = False

    by = WIN_H + (BAR_H - BTN_H) // 2
    button_edit_rect = pygame.Rect(BTN_MARGIN, by, BTN_W, BTN_H)
    button_start_rect = pygame.Rect(BTN_MARGIN + BTN_W + BTN_MARGIN, by, BTN_W, BTN_H)
    button_stop_rect = pygame.Rect(BTN_MARGIN + 2 * (BTN_W + BTN_MARGIN), by, BTN_W, BTN_H)
    button_restart_rect = pygame.Rect(BTN_MARGIN + 3 * (BTN_W + BTN_MARGIN), by, BTN_W, BTN_H)
    button_clear_rect = pygame.Rect(BTN_MARGIN + 4 * (BTN_W + BTN_MARGIN), by, BTN_W, BTN_H)
    button_save_rect = pygame.Rect(BTN_MARGIN + 5 * (BTN_W + BTN_MARGIN), by, 88, BTN_H)
    button_load_rect = pygame.Rect(BTN_MARGIN + 5 * (BTN_W + BTN_MARGIN) + 88 + BTN_MARGIN, by, 88, BTN_H)

    def draw_game_area(source: str) -> None:
        screen.fill((255, 255, 255), (0, 0, WIN_W, WIN_H))

        if source == "edit":
            for w in walls:
                if len(w.points) >= 2:
                    pygame.draw.lines(screen, (0, 0, 0), False, w.points, w.thickness)
            if len(current_stroke) >= 2:
                pygame.draw.lines(screen, (80, 80, 80), False, current_stroke, 4)
            if police_pos is not None:
                pygame.draw.circle(screen, (0, 0, 255), (int(police_pos[0]), int(police_pos[1])), 10)
                end = (police_pos[0] + math.cos(police_dir) * 25, police_pos[1] + math.sin(police_dir) * 25)
                pygame.draw.line(screen, (0, 0, 180), police_pos, end, 2)
            if enemy_pos is not None:
                pygame.draw.circle(screen, (255, 0, 0), (int(enemy_pos[0]), int(enemy_pos[1])), 10)
        else:
            s = env.state
            if s is None:
                return
            for w in s.walls:
                if len(w.points) >= 2:
                    pygame.draw.lines(screen, (0, 0, 0), False, w.points, w.thickness)
            if s.police_agents:
                p = s.police_agents[0]
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

    def draw_bar() -> None:
        screen.fill((240, 240, 240), (0, WIN_H, WIN_W, BAR_H))
        for label, r in [
            ("Edit", button_edit_rect),
            ("Play", button_start_rect),
            ("Stop", button_stop_rect),
            ("Restart", button_restart_rect),
            ("Clear", button_clear_rect),
            ("Save map", button_save_rect),
            ("Load map", button_load_rect),
        ]:
            highlight = (label == "Edit" and mode == "edit") or (label == "Play" and mode == "running")
            color = (180, 180, 255) if highlight else (220, 220, 220)
            pygame.draw.rect(screen, color, r)
            pygame.draw.rect(screen, (0, 0, 0), r, 1)
            t = font.render(label, True, (0, 0, 0))
            screen.blit(t, (r.x + (r.w - t.get_width()) // 2, r.y + (r.h - t.get_height()) // 2))
        x = button_load_rect.right + BTN_MARGIN * 2
        if mode == "edit":
            hint = "L-click: police  R-click: enemy (red dot)  Ctrl+drag: draw wall"
        elif mode == "running":
            hint = "Running — police finds static enemy. Click Stop to pause."
        else:
            hint = "Stopped — Restart or Edit to change layout"
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
                    if button_edit_rect.collidepoint(mx, my):
                        mode = "edit"
                        done = False
                        truncated = False
                    elif button_start_rect.collidepoint(mx, my):
                        opts = _build_options(walls, police_pos, police_dir, enemy_pos, enemy_dir)
                        if opts is None:
                            pass
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
                    elif button_clear_rect.collidepoint(mx, my):
                        walls = []
                        current_stroke = []
                        police_pos = None
                        enemy_pos = None
                        saved_layout = None
                        mode = "edit"
                    elif button_save_rect.collidepoint(mx, my):
                        if _save_training_map(TRAINING_MAP_PATH, walls, police_pos, police_dir, enemy_pos, enemy_dir):
                            saved_layout = _build_options(walls, police_pos, police_dir, enemy_pos, enemy_dir)
                            print(f"Saved map to {TRAINING_MAP_PATH}")
                        else:
                            print("Place police and enemy first, then Save map")
                    elif button_load_rect.collidepoint(mx, my):
                        walls, police_pos, police_dir, enemy_pos, enemy_dir = _load_training_map(TRAINING_MAP_PATH)
                        if police_pos is not None or enemy_pos is not None or walls:
                            current_stroke = []
                            if police_pos is not None and enemy_pos is not None:
                                saved_layout = _build_options(walls, police_pos, police_dir, enemy_pos, enemy_dir)
                            print(f"Loaded map from {TRAINING_MAP_PATH}")
                        else:
                            print(f"No map at {TRAINING_MAP_PATH}")
                elif mode == "edit" and my < WIN_H:
                    mods = pygame.key.get_mods()
                    if event.button == 1:
                        if mods & pygame.KMOD_CTRL:
                            current_stroke = [(mx, my)]
                        else:
                            police_pos = (float(mx), float(my))
                    elif event.button == 3:
                        enemy_pos = (float(mx), float(my))

            if event.type == pygame.MOUSEMOTION and mode == "edit" and pygame.mouse.get_pressed()[0]:
                if pygame.key.get_mods() & pygame.KMOD_CTRL and current_stroke:
                    current_stroke.append(event.pos)

            if event.type == pygame.MOUSEBUTTONUP and mode == "edit" and event.button == 1 and current_stroke:
                if len(current_stroke) >= 2:
                    walls.append(WallStroke(points=list(current_stroke)))
                current_stroke = []

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
