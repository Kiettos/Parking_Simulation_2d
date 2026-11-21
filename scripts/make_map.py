"""Sinh map JSON cho ParkSim2D từ bộ ký tự tiles & objects.

Chạy: python scripts/make_map.py
Kết quả: maps/vuong_goc_9x9.json
"""

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CELL = 30        # 30x30 px / ô -> world 270 x 270
OBJ_SIZE = 15    # 15x15 px cho Bush (B) và Lamp (L)
OBJ_OFFSET = (CELL - OBJ_SIZE) / 2.0  # 7.5 px để căn giữa ô 30x30

# Dữ liệu map nguồn theo bộ ký tự
SOURCE_DATA = {
    "maps": [
        {
            "name": "VuongGoc",
            "displayName": "Dạng 1 — Vuông góc 9x9 (Viền chữ U)",
            "cellSize": 6.0,
            "tiles": [
                "GGGGGGGGG",
                "GCCCCCCCG",
                "GCPPCPPCG",
                "GCCCCCCCG",
                "GCCCCCCCG",
                "GCPPCPPCG",
                "GCPPCPPCG",
                "HCCCCCCCH",
                "RRRRRRRRR",
            ],
            "objects": [
                "BBBBBBBBB",
                "B-------B",
                "B-------B",
                "B-------B",
                "B-------B",
                "B-------B",
                "B-------B",
                "L-------L",
                "---------",
            ],
        }
    ]
}

# Quy định bảng màu & thuộc tính để renderer sử dụng
THEME_CONFIG = {
    "tiles": {
        "C": {"name": "casual", "color": "#9e9e9e"},
        "G": {"name": "planting", "color": "#795548"},
        "H": {"name": "sidewalk", "color": "#fbc02d"},
        "R": {"name": "road", "color": "#37474f"},
        "P": {"name": "parking", "color": "#455a64"},
    },
    "objects": {
        "B": {"name": "bush", "size": OBJ_SIZE, "color": "#1976d2"},
        "L": {"name": "lamp", "size": OBJ_SIZE, "color": "#00e5ff"},
    },
    "car_color": "#0d47a1",
}


def build_map(map_cfg: dict) -> dict:
    tiles = map_cfg["tiles"]
    objects = map_cfg["objects"]

    rows = len(tiles)
    cols = len(tiles[0])

    layout = []
    obstacles = []
    slots = []

    for r in range(rows):
        row_chars = []
        for c in range(cols):
            t_char = tiles[r][c]
            o_char = objects[r][c]

            x0 = c * CELL
            y0 = r * CELL
            cx = x0 + CELL / 2.0
            cy = y0 + CELL / 2.0

            # 1. Chuyển đổi tiles sang layout chuẩn của ParkSim2D ('#' = tường/cấm vào, '.' = đường đi)
            if t_char in ("G", "H"):
                row_chars.append("#")
            else:
                row_chars.append(".")

            # 2. Tạo slots tự động tại các ô 'P'
            if t_char == "P":
                slots.append(
                    {
                        "cx": cx,
                        "cy": cy,
                        "w": 22.0,          # Rộng 22px (dư so với bề ngang xe 9.2px)
                        "h": 28.0,          # Dài 28px (dư so với chiều dài xe 21.8px)
                        "theta": math.pi / 2,  # Hướng đỗ dọc (+90 độ)
                    }
                )

            # 3. Tạo obstacles 15x15px căn giữa ô cho 'B' (Bush) và 'L' (Lamp)
            if o_char in ("B", "L"):
                obstacles.append(
                    {
                        "type": "bush" if o_char == "B" else "lamp",
                        "code": o_char,
                        "x": x0 + OBJ_OFFSET,
                        "y": y0 + OBJ_OFFSET,
                        "w": OBJ_SIZE,
                        "h": OBJ_SIZE,
                        "color": THEME_CONFIG["objects"][o_char]["color"],
                    }
                )

        layout.append("".join(row_chars))

    return {
        "name": map_cfg["name"],
        "displayName": map_cfg["displayName"],
        "grid": {
            "cols": cols,
            "rows": rows,
            "cell_size": CELL,
        },
        "theme": THEME_CONFIG,
        "tiles_raw": tiles,      # Giữ lại lưới gốc để renderer tô màu C, G, H, R, P
        "objects_raw": objects,  # Giữ lại lưới object gốc
        "car": {
            "x": 45.0,           # Đặt đầu đường Road (cột 1, hàng 8)
            "y": 8 * CELL + CELL / 2.0,  # y = 255.0 (giữa làn đường R)
            "theta": 0.0,        # Hướng sang phải (0 rad)
            "length": 21.8,      # Chiều dài (Z)
            "width": 9.2,        # Chiều ngang (X)
            "wheelbase": 14.0,   # Chiều dài cơ sở tỉ lệ theo chiều dài 21.8px
            "color": "#0d47a1",
        },
        "layout": layout,
        "obstacles": obstacles,
        "slots": slots,
    }


def main():
    for map_cfg in SOURCE_DATA["maps"]:
        data = build_map(map_cfg)
        file_slug = map_cfg["name"].lower()
        out = ROOT / "maps" / f"{file_slug}_9x9.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        print(
            f"Wrote {out}  ({data['grid']['rows']}x{data['grid']['cols']}, "
            f"{len(data['slots'])} slots, {len(data['obstacles'])} obstacles)"
        )


if __name__ == "__main__":
    main()