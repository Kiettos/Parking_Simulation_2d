import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mapark.map_loader import world_from_map
from mapark.tile_map import build_runtime_config
from render2d.app import run


def main():
    parser = argparse.ArgumentParser(description="Hybrid A* + Pure Pursuit demo")
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
    args = parser.parse_args()

    cfg = build_runtime_config(args.map_config, args.map, args.map_name)
    world = world_from_map(cfg["map_spec"])
    run(world)


if __name__ == "__main__":
    main()
