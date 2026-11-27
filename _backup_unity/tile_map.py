"""Nạp 2 file JSON: config tile/object/agent + map 21x21.

- configs/parking_map_config.json : tileSize, objectSize, agent, legend màu
- maps/map_and_objects_11.json    : danh sách maps (tiles + objects)

Chuyển trực tiếp thành spec pixel OBB cho mapark.map_loader.world_from_map,
không sinh file trung gian.
"""

import json
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_TRAINING = {
    "n_envs": 4,
    "total_timesteps": 500_000,
    "max_steps": 400,
    "dt": 0.15,
    "decoy_count": 5,
    "decoy_seed": 7,
    "decoy_success_target": 20,
    "random_target_slot": True,
    "success_dist": 8.0,
    "success_theta_deg": 20.0,
    "sensor_range": 150.0,
    "obs_dist_scale": 890.0,
    # Reward va chạm chung + theo từng loại tile/object
    "collision_reward": -10.0,
    "collision_rewards": {
        "tile:G": -10.0,     # planting
        "tile:H": -10.0,     # sidewalk
        "object:B": -6.0,    # bush
        "object:L": -6.0,    # lamp
        "decoy": -10.0,
    },
    "terminate_on_collision": True,
    "start_jitter": [3.0, 3.0, 0.2],
    "max_speed_fwd": 60.0,
    "max_speed_rev": 30.0,
    "accel": 120.0,
    "decel": 100.0,
    "log_every_steps": 10_000,
    "model_out": "models/ppo_park_tiles",
}

DEFAULT_PPO = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 256,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "net_arch": [64, 64],
}


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def load_json(path):
    with resolve(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def pick_map(map_doc, name=None):
    maps = map_doc.get("maps") or []
    if not maps:
        raise ValueError("Map JSON không có 'maps'")
    if name:
        found = next((m for m in maps if m.get("name") == name), None)
        if found is None:
            names = ", ".join(m.get("name", "?") for m in maps)
            raise ValueError(f"Không có map '{name}' (có: {names})")
        return found
    return maps[0]


def build_map_spec(map_config, map_data, training=None):
    """Tile/object chars -> spec pixel OBB cho world_from_map."""
    training = training or DEFAULT_TRAINING
    ts = float(map_config["tileSize"])
    obj_size = float(map_config.get("objectSize", 30))

    tiles = [str(r) for r in map_data["tiles"]]
    objects = [str(r) for r in (map_data.get("objects") or [])]
    rows, cols = len(tiles), len(tiles[0])
    if any(len(r) != cols for r in tiles):
        raise ValueError("Các hàng tiles phải bằng nhau")

    tile_legend = map_config["legend"]["tiles"]
    object_legend = map_config["legend"]["objects"]

    layout = []
    layout_codes = []
    slots = []
    obstacles = []

    for r, row in enumerate(tiles):
        line = []
        run_codes = []
        for c, ch in enumerate(row):
            if ch not in tile_legend:
                raise ValueError(f"Tile '{ch}' tại ({r},{c}) không có trong legend")
            spec = tile_legend[ch]
            walkable = bool(spec.get("walkable", True))
            line.append("." if walkable else "#")
            run_codes.append("." if walkable else ch)

            cx = c * ts + ts / 2.0
            cy = r * ts + ts / 2.0

            if spec.get("isGoal") or ch == "P":
                slots.append(
                    {
                        "cx": cx,
                        "cy": cy,
                        "w": 44.0,
                        "h": 56.0,
                        "theta": math.pi / 2,
                    }
                )

            obj_ch = objects[r][c] if r < len(objects) and c < len(objects[r]) else "-"
            if obj_ch != "-":
                if obj_ch not in object_legend:
                    raise ValueError(
                        f"Object '{obj_ch}' tại ({r},{c}) không có trong legend"
                    )
                ospec = object_legend[obj_ch]
                # Object nằm trên tile không đi được thì tile đã là vật cản;
                # bỏ qua để giảm số OBB (tránh chậm raycast/planner).
                if ospec.get("isObstacle") and walkable:
                    obstacles.append(
                        {
                            "type": ospec.get("name", obj_ch),
                            "code": obj_ch,
                            "kind": f"object:{obj_ch}",
                            "cx": cx,
                            "cy": cy,
                            "w": obj_size,
                            "h": obj_size,
                        }
                    )
        layout.append("".join(line))
        row_codes = []
        row_codes.append("".join(run_codes))
        layout_codes.append("".join(row_codes))
    agent = map_config.get("agent") or {}
    length = float(agent.get("length", 18.4))
    width = float(agent.get("width", 43.6))

    # Spawn: object 'S' nếu có, không thì ô drivable hàng cuối (ưu tiên cột 1).
    sx = sy = None
    for r, row in enumerate(objects):
        c = row.find("S")
        if c >= 0:
            sx, sy = c * ts + ts / 2.0, r * ts + ts / 2.0
            break
    if sx is None:
        spawn = map_data.get("spawn")
        if spawn:
            sx = float(spawn["col"]) * ts + ts / 2.0
            sy = float(spawn["row"]) * ts + ts / 2.0
    if sx is None:
        for c in (1, cols // 2, 0):
            if layout[rows - 1][c] == ".":
                sx = c * ts + ts / 2.0
                sy = (rows - 1) * ts + ts / 2.0
                break
    if sx is None:
        raise ValueError("Không tìm được điểm spawn trên map")

    return {
        "name": map_data.get("name", "map"),
        "grid": {"cols": cols, "rows": rows, "cell_size": ts},
        "layout": layout,
        "layout_codes": layout_codes,
        "obstacles": obstacles,
        "slots": slots,
        "tiles_raw": tiles,
        "objects_raw": objects,
        "legend": map_config["legend"],
        "tile_size": ts,
        "obj_size": obj_size,
        "car": {
            "x": sx,
            "y": sy,
            "theta": 0.0,
            "length": length,
            "width": width,
            "wheelbase": float(agent.get("wheelbase", round(length * 0.64, 2))),
            "color": agent.get("color", "#0d47a1"),
            "max_speed_fwd": float(training["max_speed_fwd"]),
            "max_speed_rev": float(training["max_speed_rev"]),
            "accel": float(training["accel"]),
            "decel": float(training["decel"]),
        },
    }


def build_runtime_config(
    map_config_path="configs/parking_map_config.json",
    map_path="maps/map_and_objects_11.json",
    map_name=None,
    training=None,
    ppo=None,
):
    """Config runtime cho ParkingEnv từ đúng 2 file JSON."""
    map_config = load_json(map_config_path)
    map_doc = load_json(map_path)
    map_data = pick_map(map_doc, map_name)

    tr = dict(DEFAULT_TRAINING)
    tr.update(map_config.get("training") or {})
    tr.update(training or {})
    pp = dict(DEFAULT_PPO)
    pp.update(map_config.get("ppo") or {})
    pp.update(ppo or {})

    spec = build_map_spec(map_config, map_data, tr)
    return {
        "map_spec": spec,
        "map_name": spec["name"],
        "training": tr,
        "ppo": pp,
    }
