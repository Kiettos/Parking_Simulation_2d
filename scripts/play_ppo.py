import argparse
import sys
from pathlib import Path

import pygame

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO

from mapark.rl_env import ParkingEnv
from mapark.sensors import sense_all
from mapark.tile_map import build_runtime_config
from render2d.app import draw_world

# Mức tăng tốc: số bước môi trường cho mỗi khung hình
SPEED_STEPS = [1, 2, 4, 8, 16, 32]
FPS = 30


def main():
    parser = argparse.ArgumentParser(description="Run a trained PPO agent")
    parser.add_argument(
        "--map-config",
        default="configs/parking_map_config.json",
        help="Tile/object/agent config JSON",
    )
    parser.add_argument(
        "--map",
        default="configs/maps.json",
        help="Map JSON (21x21 tiles/objects)",
    )
    parser.add_argument("--map-name", default=None, help="Map name in JSON")
    parser.add_argument(
        "--model",
        default=None,
        help="Path to model .zip (default: training.model_out)",
    )
    parser.add_argument(
        "--speed",
        type=int,
        default=1,
        choices=SPEED_STEPS,
        help="Steps per frame at startup",
    )
    args = parser.parse_args()

    cfg = build_runtime_config(args.map_config, args.map, args.map_name)
    tr = cfg["training"]
    model_path = Path(args.model or tr.get("model_out", "models/ppo_park_tiles"))
    if not model_path.is_absolute():
        model_path = ROOT / model_path

    env = ParkingEnv(cfg)
    model = PPO.load(str(model_path), device="cpu")

    # Model cũ train với Discrete(6) không chạy được trên env Box(3,) và ngược lại
    model_act = model.action_space
    if getattr(model_act, "shape", None) != getattr(env.action_space, "shape", None):
        print(
            f"[warn] model action_space={model_act} khác env={env.action_space}."
            " Phải train lại model với action space hiện tại."
        )

    pygame.init()
    screen = pygame.display.set_mode(
        (int(env.world.width), int(env.world.height))
    )
    pygame.display.set_caption(
        f"ParkSim2D — PPO ({model_path.name})"
        "  [SPACE] pause  [↑/↓] speed  [N] no-render  [R] reset  [ESC] quit"
    )
    font = pygame.font.SysFont("consolas", 16)
    clock = pygame.time.Clock()

    obs, info = env.reset()
    total_reward = 0.0
    status = "Running"
    last_result = ""

    speed_idx = SPEED_STEPS.index(args.speed)
    paused = False
    render_world = True
    episodes_done = 0
    successes = 0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0.0
                    status = "Reset"
                    last_result = ""
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key in (pygame.K_UP, pygame.K_RIGHT, pygame.K_EQUALS, pygame.K_PLUS):
                    speed_idx = min(speed_idx + 1, len(SPEED_STEPS) - 1)
                elif event.key in (pygame.K_DOWN, pygame.K_LEFT, pygame.K_MINUS):
                    speed_idx = max(speed_idx - 1, 0)
                elif event.key == pygame.K_n:
                    render_world = not render_world

        steps = 0 if paused else SPEED_STEPS[speed_idx]

        ended = False
        for _ in range(steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                episodes_done += 1
                if info.get("is_success"):
                    successes += 1
                    status = "PARKED"
                elif info.get("is_collision"):
                    status = "COLLIDED"
                else:
                    status = "Timeout"
                last_result = f"return={total_reward:.1f}"
                obs, info = env.reset()
                total_reward = 0.0
                ended = True
                if steps == 1:
                    pygame.time.wait(700)
                break

        clock.tick(FPS if SPEED_STEPS[speed_idx] < 8 else 0)

        if not render_world or SPEED_STEPS[speed_idx] >= 16:
            continue

        env.world.sensors = sense_all(
            env.world, env.car, max_range=env.sensor_range
        )
        draw_world(screen, env.world)

        hud1 = font.render(
            f"reward={total_reward:.2f}  slot={info.get('target_slot')}"
            f"  ep={info.get('episode_index')}  layout={info.get('layout_index')}"
            f"  layout_succ={info.get('layout_successes')}"
            f"  total_succ={info.get('total_successes')}",
            True,
            (30, 30, 30),
        )
        screen.blit(hud1, (10, 10))
        hud2 = font.render(f"{status}  {last_result}", True, (30, 30, 30))
        screen.blit(hud2, (10, 30))
        hud3 = font.render(
            f"x{SPEED_STEPS[speed_idx]} steps/frame"
            f"  episodes={episodes_done}  parked={successes}"
            f"  {'PAUSED' if paused else ''}"
            f"  {'(no render)' if not render_world else ''}",
            True,
            (0, 90, 190),
        )
        screen.blit(hud3, (10, 50))
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
