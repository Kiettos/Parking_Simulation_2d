import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mapark.car import BicycleCar
from mapark.geometry import OBB
from mapark.world import World
from render2d.app import run


def load_config(path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_world(cfg):
    w = cfg["world"]
    world = World(w["width"], w["height"])

    c = cfg["car"]
    sx, sy, st = c["start"]
    world.car = BicycleCar(
        sx, sy, st,
        length=c.get("length", 44),
        width=c.get("width", 20),
        wheelbase=c.get("wheelbase", 28),
    )
    for o in cfg.get("obstacles", []):
        world.add_obstacle(OBB(o["cx"], o["cy"], o["w"], o["h"], o.get("theta", 0.0)))
    for s in cfg.get("slots", []):
        world.add_slot(OBB(s["cx"], s["cy"], s["w"], s["h"], s.get("theta", 0.0)))
    return world


def main():
    cfg = load_config(ROOT / "configs" / "default.yaml")
    world = build_world(cfg)
    run(world)


if __name__ == "__main__":
    main()