import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from mapark.callbacks import RewardStatsCallback, TargetReturnCallback
from mapark.rl_env import ParkingEnv
from mapark.tile_map import build_runtime_config


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else (ROOT / p)


def main():
    parser = argparse.ArgumentParser(description="Train PPO for ParkSim2D")
    parser.add_argument(
        "--map-config",
        default="configs/parking_map_config.json",
        help="Tile/object/agent config JSON",
    )
    parser.add_argument(
        "--map",
        default="maps/map_and_objects_11.json",
        help="Map JSON (21x21 tiles/objects)",
    )
    parser.add_argument("--map-name", default=None, help="Map name in JSON")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--n-envs", type=int, default=None)
    parser.add_argument("--model-out", default=None)
    args = parser.parse_args()

    cfg = build_runtime_config(args.map_config, args.map, args.map_name)
    spec = cfg["map_spec"]
    tr = cfg["training"]
    ppo_cfg = cfg["ppo"]

    n_envs = args.n_envs or int(tr.get("n_envs", 4))
    total_timesteps = args.timesteps or int(tr.get("total_timesteps", 500_000))
    log_every = int(tr.get("log_every_steps", 10_000))

    print(
        f"[map] {cfg['map_name']}: {spec['grid']['rows']}x{spec['grid']['cols']}"
        f" cell={spec['grid']['cell_size']}"
        f" slots={len(spec['slots'])} obstacles={len(spec['obstacles'])}"
        f" car={spec['car']['length']}x{spec['car']['width']}",
        flush=True,
    )

    # Windows: 1 env dùng DummyVecEnv cho chắc; nhiều env dùng SubprocVecEnv
    vec_cls = DummyVecEnv if n_envs == 1 else SubprocVecEnv
    env = make_vec_env(
        lambda: ParkingEnv(cfg),
        n_envs=n_envs,
        vec_env_cls=vec_cls,
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=float(ppo_cfg.get("learning_rate", 3e-4)),
        n_steps=int(ppo_cfg.get("n_steps", 2048)),
        batch_size=int(ppo_cfg.get("batch_size", 256)),
        n_epochs=int(ppo_cfg.get("n_epochs", 10)),
        gamma=float(ppo_cfg.get("gamma", 0.99)),
        gae_lambda=float(ppo_cfg.get("gae_lambda", 0.95)),
        clip_range=float(ppo_cfg.get("clip_range", 0.2)),
        ent_coef=float(ppo_cfg.get("ent_coef", 0.01)),
        policy_kwargs=dict(net_arch=list(ppo_cfg.get("net_arch", [64, 64]))),
        tensorboard_log=str(ROOT / "runs"),
    )

    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)

    callback_list = [
        CheckpointCallback(
            save_freq=20_000,
            save_path=str(models_dir),
            name_prefix="ppo_park",
        ),
        RewardStatsCallback(log_every=log_every),
    ]

    # Chỉ dừng sớm theo mean return nếu config đặt target_return.
    target_return = tr.get("target_return")
    if target_return is not None:
        callback_list.append(
            TargetReturnCallback(
                target_return=float(target_return),
                window=int(tr.get("target_window", 100)),
                extra_episodes=int(tr.get("extra_episodes", 100)),
            )
        )

    callbacks = CallbackList(callback_list)

    print(
        f"[train] envs={n_envs} total_timesteps={total_timesteps}"
        f" log_every={log_every}"
        f" decoys={tr.get('decoy_count', 0)}"
        f" stages: spawn/{tr.get('spawn_change_every', 20)}"
        f" target/{tr.get('target_change_every', 60)}"
        f" decoy/{tr.get('decoy_change_every', 180)}",
        flush=True,
    )

    model.learn(total_timesteps=total_timesteps, callback=callbacks)

    out = resolve(args.model_out or tr.get("model_out", "models/ppo_park_tiles"))
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out))
    print(f"[train] saved -> {out}.zip", flush=True)


if __name__ == "__main__":
    main()
