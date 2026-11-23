# ParkSim2D — xe tự đỗ 2D (Hybrid A* + PPO)

Mô phỏng 2D bài toán xe tự đỗ, gồm 2 hướng tiếp cận:

- **Hybrid A\*** sinh đường đi + **Pure Pursuit** bám đường (demo trực quan).
- **PPO / Stable-Baselines3** học lái trực tiếp từ cảm biến (13 chiều quan sát).

## Tính năng mới (bản map 21×21)

1. **Map nạp từ JSON 21×21** (`maps/parking_21x21.json`): lưới tường, 10 ô đỗ,
   vị trí xe xuất phát, vật cản — tất cả định nghĩa trong JSON.
2. **Xe decoy theo success**: ban đầu bãi trống; episode đỗ thành công đúng
   hướng đầu tiên sẽ spawn `decoy_count` (5) xe vào các slot ngẫu nhiên. Sau đó
   đủ `decoy_success_target` (20) episode đỗ thành công thì đổi layout decoy
   (spawn lại ngẫu nhiên). Target đổi mỗi episode; decoy giữ nguyên giữa các
   episode.
3. **In thống kê mỗi 10.000 step**: mean/std reward theo step, mean/std episode
   return, success rate (số episode kết thúc trong cửa sổ).
4. **Không dừng sớm** (mặc định): train đến `total_timesteps`. Có thể bật lại
   `target_return` + `extra_episodes` nếu muốn dừng theo mean return.

## Cài đặt

```powershell
# Python 3.11 (đã có stable-baselines3)
.venv311\Scripts\python.exe -m pip install -r requirements.txt
```

## Chạy

```powershell
# Train trên map 21x21 (mặc định)
.venv311\Scripts\python.exe scripts\train_ppo.py

# Train ít bước để thử
.venv311\Scripts\python.exe scripts\train_ppo.py --timesteps 12000 --n-envs 1

# Chạy agent đã train
.venv311\Scripts\python.exe scripts\play_ppo.py --model models\ppo_park_21x21

# Xuất PPO sang ONNX
.venv311\Scripts\python.exe scripts\export_onnx.py --model models\ppo_park_final --output models\ppo_park_final.onnx

# Demo Hybrid A* + Pure Pursuit (SPACE = plan, ENTER = follow, D = decoy, R = reset)
.venv311\Scripts\python.exe scripts\run_demo.py

# Sinh lại map JSON
.venv311\Scripts\python.exe scripts\make_map.py
```

Lưu ý Windows: `SubprocVecEnv` cần chạy trong terminal thường (không bị sandbox
chặn named pipe). Nếu lỗi, dùng `--n-envs 1` (tự chuyển sang `DummyVecEnv`).

## Định dạng map JSON

```jsonc
{
  "grid": { "cols": 21, "rows": 21, "cell_size": 40 },  // world = 840x840 px
  "car": { "x": 100, "y": 720, "theta": 0.0,
           "length": 44, "width": 20, "wheelbase": 28 },
  "layout": ["#####################", "..."],           // '#' tường, '.' trống
  "obstacles": [ { "cx": 300, "cy": 150, "w": 140, "h": 40, "theta": 0.0 } ],
  "slots": [ { "cx": 520, "cy": 220, "w": 28, "h": 60, "theta": 1.5708 } ]
}
```

- Toạ độ tính bằng pixel, gốc trên-trái.
- Tường ở viền lưới được bỏ qua (World đã chặn biên); tường bên trong được gộp
  theo hàng thành OBB để raycast nhanh.
- `slots` là các ô đỗ; xe decoy được đặt đúng tâm slot, cùng hướng slot.

## Cấu hình chính (`configs/parking_21x21.yaml`)

| Khoá | Ý nghĩa |
|---|---|
| `map` | Đường dẫn map JSON |
| `training.decoy_count` | Số xe decoy (5) |
| `training.decoy_success_target` | Số success để đổi layout decoy (20) |
| `training.decoy_seed` | Seed cho layout decoy |
| `training.random_target_slot` | Chọn đích ngẫu nhiên trong slot trống (map mới: `true`; config cũ mặc định `false` = slot 0) |
| `training.success_dist` | Khoảng cách tâm xe ↔ tâm slot để tính success (15 px) |
| `training.success_theta_deg` | Sai lệch hướng tối đa để tính success (20°) |
| `training.start_jitter` | Random hoá vị trí/hướng xuất phát |
| `training.log_every_steps` | Chu kỳ in mean/std (10.000) |
| `training.target_return` | (Tuỳ chọn) mean return để dừng sớm; bỏ trống = train đủ |
| `training.target_window` | Số episode lấy trung bình khi dùng `target_return` |
| `training.extra_episodes` | Số episode chạy thêm sau khi đạt target return |
| `training.model_out` | Đường dẫn model lưu cuối |
| `ppo.*` | Siêu tham số PPO |

## Cấu trúc

```
mapark/
  car.py            # bicycle model
  world.py          # world, slot, decoy, va chạm
  sensors.py        # 8 tia cảm biến ego-centric
  map_loader.py     # nạp map JSON -> World
  rl_env.py         # Gymnasium env (13 obs, 6 action)
  callbacks.py      # in thống kê, log mốc spawn/respawn decoy
  path_follower.py  # pure pursuit
render2d/app.py     # vẽ map/sensor/path + demo
scripts/            # train / play / demo / make_map
configs/            # default.yaml (kiểu cũ), parking_21x21.yaml
maps/               # map JSON 21x21
```
