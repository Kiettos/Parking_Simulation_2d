import sys
from pathlib import Path

import pygame
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO

from mapark.rl_env import ParkingEnv
from mapark.sensors import sense_all
from render2d.app import draw_world


def main():
    with (ROOT / "configs" / "default.yaml").open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    env = ParkingEnv(cfg)
    model = PPO.load(str(ROOT / "models" / "ppo_park_final"))

    pygame.init()
    screen = pygame.display.set_mode(
        (int(env.world.width), int(env.world.height))
    )
    pygame.display.set_caption("ParkSim2D — PPO agent")
    font = pygame.font.SysFont("consolas", 16)
    clock = pygame.time.Clock()

    obs, _ = env.reset()
    total_reward = 0.0
    status = "Running"

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    obs, _ = env.reset()
                    total_reward = 0.0
                    status = "Reset"

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(int(action))
        total_reward += reward

        env.world.sensors = sense_all(env.world, env.car, max_range=180.0)
        draw_world(screen, env.world)

        hud1 = font.render(f"reward={total_reward:.2f}  {status}", True, (30, 30, 30))
        screen.blit(hud1, (10, 10))
        pygame.display.flip()
        clock.tick(30)   # hiện tại 8; chậm lại để quan sát

        if terminated or truncated:
            if reward >= 100:
                status = "PARKED ✓"
            elif reward <= -5:
                status = "COLLIDED"
            else:
                status = "Timeout"
            pygame.time.wait(1000)
            obs, _ = env.reset()
            total_reward = 0.0

    pygame.quit()


if __name__ == "__main__":
    main()