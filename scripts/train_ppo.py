import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from mapark.rl_env import ParkingEnv


def load_config():
    with (ROOT / "configs" / "default.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()
    n_envs = 4

    env = make_vec_env(
        lambda: ParkingEnv(cfg),
        n_envs=n_envs,
        vec_env_cls=SubprocVecEnv,   # Windows đôi khi lỗi → đổi sang DummyVecEnv
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        # Shared trunk 64→64, rồi tách actor/critic head
        policy_kwargs=dict(net_arch=[64, 64]),
        tensorboard_log=str(ROOT / "runs"),
    )

    (ROOT / "models").mkdir(exist_ok=True)
    checkpoint_cb = CheckpointCallback(
        save_freq=20000,
        save_path=str(ROOT / "models"),
        name_prefix="ppo_park",
    )

    model.learn(total_timesteps=500_000, callback=checkpoint_cb)
    model.save(str(ROOT / "models" / "ppo_park_final"))
    print("Training done → models/ppo_park_final.zip")


if __name__ == "__main__":
    main()