"""Nạp map 2D từ file JSON (lưới cols x rows + các object OBB).

Định dạng JSON (xem maps/parking_21x21.json):

{
  "name": "parking_lot_21x21",
  "grid": {"cols": 21, "rows": 21, "cell_size": 40},
  "car": {"x": 100, "y": 700, "theta": 0.0,
          "length": 44, "width": 20, "wheelbase": 28},
  "layout": ["#####################", ...],   # tuỳ chọn: '#' = tường, '.' = trống
  "obstacles": [{"cx": 300, "cy": 150, "w": 140, "h": 40, "theta": 0.0}],
  "slots":     [{"cx": 520, "cy": 260, "w": 60,  "h": 28, "theta": 0.0}, ...]
}

Toạ độ tính bằng pixel. World rộng = cols * cell_size, cao = rows * cell_size.
Tường ở viền lưới được bỏ qua vì World đã chặn biên; tường bên trong được
gộp thành các OBB theo hàng để giảm số vật cản khi raycast.
"""

import json
from pathlib import Path

from Unimportant.geometry import OBB

from mapark.car import BicycleCar
from mapark.world import World

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_map_path(path):
    """Đường dẫn tương đối được hiểu theo gốc project."""
    p = Path(path)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def load_map(path):
    """Đọc + validate file map JSON."""
    p = resolve_map_path(path)
    with p.open("r", encoding="utf-8") as f:
        spec = json.load(f)
    validate_map(spec)
    return spec


def validate_map(spec):
    for key in ("grid", "car", "slots"):
        if key not in spec:
            raise ValueError(f"Map JSON thiếu khoá '{key}'")

    grid = spec["grid"]
    cols, rows = int(grid["cols"]), int(grid["rows"])
    cell = float(grid.get("cell_size", 40))
    if cols <= 0 or rows <= 0:
        raise ValueError("grid.cols / grid.rows phải > 0")
    if cell <= 0:
        raise ValueError("grid.cell_size phải > 0")

    layout = spec.get("layout")
    if layout is not None:
        if len(layout) != rows:
            raise ValueError(
                f"layout có {len(layout)} hàng, cần đúng {rows}"
            )
        for r, line in enumerate(layout):
            if len(line) != cols:
                raise ValueError(
                    f"layout hàng {r} có {len(line)} cột, cần đúng {cols}"
                )
            bad = set(line) - {".", "#"}
            if bad:
                raise ValueError(
                    f"layout hàng {r} chứa ký tự lạ {sorted(bad)}; chỉ dùng '.' và '#'"
                )

    if not spec["slots"]:
        raise ValueError("Map JSON cần ít nhất 1 slot")

    for i, slot in enumerate(spec["slots"]):
        missing = {"cx", "cy", "w", "h"} - set(slot)
        if missing:
            raise ValueError(f"slot #{i} thiếu {sorted(missing)}")

    car = spec["car"]
    missing = {"x", "y"} - set(car)
    if missing:
        raise ValueError(f"car thiếu {sorted(missing)}")


def _add_walls(world, layout, cell):
    """Gộp ô '#' thành các OBB chữ nhật (ngang + dọc)."""
    rows, cols = len(layout), len(layout[0])

    runs = []
    for r in range(rows):
        c = 0
        while c < cols:
            if layout[r][c] != "#":
                c += 1
                continue
            c0 = c
            while c < cols and layout[r][c] == "#":
                c += 1
            runs.append((r, c0, c - 1))

    run_set = set(runs)
    used = set()
    for r, c0, c1 in runs:
        if (r, c0, c1) in used:
            continue
        used.add((r, c0, c1))
        r2 = r
        while (r2 + 1, c0, c1) in run_set and (r2 + 1, c0, c1) not in used:
            r2 += 1
            used.add((r2, c0, c1))

        n = c1 - c0 + 1
        h = r2 - r + 1
        cx = c0 * cell + n * cell / 2.0
        cy = r * cell + h * cell / 2.0
        world.add_wall(OBB(cx, cy, n * cell, h * cell, 0.0))


def world_from_map(spec):
    """Dựng World + xe từ spec JSON đã validate."""
    grid = spec["grid"]
    cols, rows = int(grid["cols"]), int(grid["rows"])
    cell = float(grid.get("cell_size", 40))

    world = World(cols * cell, rows * cell)
    world.cols = cols
    world.rows = rows
    world.cell_size = cell

    car = spec["car"]
    world.car = BicycleCar(
        float(car["x"]),
        float(car["y"]),
        float(car.get("theta", 0.0)),
        length=float(car.get("length", 44)),
        width=float(car.get("width", 20)),
        wheelbase=float(car.get("wheelbase", 28)),
    )
    world.tiles_raw = spec.get("tiles_raw")
    world.objects_raw = spec.get("objects_raw")
    world.legend = spec.get("legend")
    world.obj_size = float(spec.get("obj_size", 15))
    world.car_color = car.get("color")
    # Cho phép map JSON override động học xe (map nhỏ cần tốc độ nhỏ hơn).
    for key in ("max_speed_fwd", "max_speed_rev", "accel", "decel"):
        if key in car:
            setattr(world.car, key, float(car[key]))
    world.start_pose = (world.car.x, world.car.y, world.car.theta)

    layout = spec.get("layout")
    if layout:
        _add_walls(world, layout, cell)

    for o in spec.get("obstacles", []):
        cx = o.get("cx")
        cy = o.get("cy")
        if cx is None or cy is None:
            # Hỗ trợ cả dạng x/y/w/h (góc trên-trái) do make_map.py xuất ra.
            cx = float(o.get("x", 0.0)) + float(o["w"]) / 2.0
            cy = float(o.get("y", 0.0)) + float(o["h"]) / 2.0
        world.add_obstacle(
            OBB(cx, cy, o["w"], o["h"], o.get("theta", 0.0))
        )
    for s in spec["slots"]:
        world.add_slot(
            OBB(s["cx"], s["cy"], s["w"], s["h"], s.get("theta", 0.0))
        )

    return world
