import math
import random

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from mapark.car import BicycleCar
from Unimportant.geometry import OBB
from mapark.map_loader import load_map, world_from_map
from mapark.sensors import sense_all
from mapark.world import World


class ParkingEnv(gym.Env):
    metadata = {"render_modes": []}

    SENSOR_ORDER = ["F", "FR", "R", "BR", "B", "BL", "L", "FL"]

    def __init__(self, config, max_steps=400, dt=0.15):
        super().__init__()
        self.config = config
        self.train_cfg = dict(config.get("training", {}))
        self.max_steps = int(self.train_cfg.get("max_steps", max_steps))
        self.dt = float(self.train_cfg.get("dt", dt))

        # --- Map: ưu tiên spec dựng từ 2 file JSON, fallback map JSON/config cũ ---
        if config.get("map_spec"):
            self.world = world_from_map(config["map_spec"])
            self.start_pose = self.world.start_pose
        elif config.get("map"):
            self.world = world_from_map(load_map(config["map"]))
            self.start_pose = self.world.start_pose
        else:
            self.world = self._build_world()
            c = config["car"]
            self.start_pose = (
                float(c["start"][0]),
                float(c["start"][1]),
                float(c["start"][2]),
            )
        self.car = self.world.car
        self.num_slots = len(self.world.slots)
        if self.num_slots == 0:
            raise ValueError("Map cần ít nhất một slot đỗ")

        # --- Xe decoy: chỉ spawn sau khi đỗ thành công, rồi đổi layout sau
        #     `decoy_success_target` lần đỗ thành công (mặc định 20) ---
        self.decoy_count = int(self.train_cfg.get("decoy_count", 0))
        self.decoy_seed = int(self.train_cfg.get("decoy_seed", 0))
        self.decoy_success_target = max(1, int(self.train_cfg.get("decoy_success_target", 20)))
        # luôn chừa ít nhất 1 slot trống làm đích
        self.decoy_count = max(0, min(self.decoy_count, self.num_slots - 1))

        self.success_dist = float(self.train_cfg.get("success_dist", 15.0))
        self.success_theta = math.radians(
            float(self.train_cfg.get("success_theta_deg", 20.0))
        )
        self.sensor_range = float(self.train_cfg.get("sensor_range", 180.0))
        self.obs_dist_scale = float(self.train_cfg.get("obs_dist_scale", 800.0))

        jitter = self.train_cfg.get("start_jitter", [30.0, 30.0, 0.3])
        self.start_jitter = (float(jitter[0]), float(jitter[1]), float(jitter[2]))

        # Map mới: đích ngẫu nhiên trong các slot trống.
        # Config kiểu cũ: giữ slot 0 để không phá model đã train trước đây.
        default_random_target = "map" in config
        self.random_target_slot = bool(
            self.train_cfg.get("random_target_slot", default_random_target)
        )

        self.episode_index = 0
        self.target_index = 0
        self.decoy_spawned = False
        self.decoy_slots = []
        self.layout_index = 0
        self.layout_successes = 0
        self.total_successes = 0
        self._pending_decoy_spawn = False

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(13,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(6)

        # (throttle_dir, steer_dir): ±1
        self._actions = [
            (1, -1), (1, 0), (1, 1),
            (-1, -1), (-1, 0), (-1, 1),
        ]

        self.step_count = 0
        self.prev_dist = 0.0

    # ------------------------------------------------------------------
    # Dựng world
    # ------------------------------------------------------------------
    def _build_world(self):
        """World kiểu cũ (config YAML có world/car/obstacles/slots)."""
        w = self.config["world"]
        world = World(w["width"], w["height"])
        c = self.config["car"]
        world.car = BicycleCar(
            *c["start"],
            length=c["length"],
            width=c["width"],
            wheelbase=c["wheelbase"],
        )
        for o in self.config["obstacles"]:
            world.add_obstacle(
                OBB(o["cx"], o["cy"], o["w"], o["h"], o.get("theta", 0.0))
            )
        for s in self.config["slots"]:
            world.add_slot(
                OBB(s["cx"], s["cy"], s["w"], s["h"], s.get("theta", 0.0))
            )
        return world

    # ------------------------------------------------------------------
    # Decoy
    # ------------------------------------------------------------------
    def _spawn_decoy_layout(self):
        """Đặt `decoy_count` xe vào slot ngẫu nhiên (layout mới).

        Layout tất định theo `decoy_seed + layout_index` để tái lập được.
        """
        self.layout_successes = 0
        self._pending_decoy_spawn = False

        if self.decoy_count <= 0 or self.num_slots <= 1:
            self.world.set_decoys([])
            self.decoy_slots = []
            return

        self.layout_index += 1
        rng = random.Random(self.decoy_seed * 1_000_003 + self.layout_index)
        k = min(self.decoy_count, self.num_slots - 1)
        ids = sorted(rng.sample(range(self.num_slots), k))

        decoys = []
        for i in ids:
            slot = self.world.slots[i]
            # xe decoy đỗ đúng tâm slot, cùng hướng với slot
            decoys.append(
                OBB(slot.cx, slot.cy, self.car.length, self.car.width, slot.theta)
            )
        self.world.set_decoys(decoys)
        self.world.decoy_slots = ids
        self.decoy_slots = ids

    def force_decoy_layout(self):
        """Bật decoy ngay ở lần reset kế tiếp (dùng cho demo/eval)."""
        self.decoy_spawned = True
        self._pending_decoy_spawn = True

    def _on_success(self, info):
        """Cập nhật bộ đếm success và quyết định spawn/đổi layout decoy.

        - Success đầu tiên: spawn decoy ở episode kế tiếp.
        - Sau đó: đủ `decoy_success_target` success thì đổi layout decoy.
        """
        self.total_successes += 1

        if self.decoy_count <= 0:
            info["total_successes"] = self.total_successes
            info["decoy_spawned"] = False
            return

        if not self.decoy_spawned:
            self.decoy_spawned = True
            self.layout_successes = 0
            self._pending_decoy_spawn = True
            info["decoy_event"] = "spawn"
        else:
            self.layout_successes += 1
            if self.layout_successes >= self.decoy_success_target:
                info["decoy_event"] = "respawn"
                info["layout_completed"] = True
                info["completed_successes"] = self.layout_successes
                self._pending_decoy_spawn = True

        info["layout_index"] = self.layout_index
        info["layout_successes"] = self.layout_successes
        info["total_successes"] = self.total_successes
        info["decoy_spawned"] = self.decoy_spawned

    # ------------------------------------------------------------------
    # Quan sát / reset
    # ------------------------------------------------------------------
    def _sensors(self):
        return sense_all(self.world, self.car, max_range=self.sensor_range)

    def _get_obs(self):
        car = self.car
        sensors = self._sensors()

        sensor_vals = []
        for name in self.SENSOR_ORDER:
            dist = sensors[name][0]
            # Gần vật cản → gần 1; trống → gần 0
            sensor_vals.append(
                1.0 - min(dist, self.sensor_range) / self.sensor_range
            )

        v_norm = max(-1.0, min(1.0, car.v / car.max_speed_fwd))
        steer_norm = max(-1.0, min(1.0, car.steer / car.max_steer))

        slot = self.world.slots[self.target_index]
        dx = slot.cx - car.x
        dy = slot.cy - car.y
        c, s = math.cos(-car.theta), math.sin(-car.theta)
        lx = c * dx - s * dy
        ly = s * dx + c * dy
        dist = math.hypot(dx, dy)
        ang = math.atan2(ly, lx)

        obs = sensor_vals + [
            v_norm,
            steer_norm,
            math.cos(ang),
            math.sin(ang),
            min(1.0, dist / self.obs_dist_scale),
        ]
        return np.array(obs, dtype=np.float32)

    def _spawn_car(self):
        base_x, base_y, base_theta = self.start_pose
        jx, jy, jt = self.start_jitter
        for _ in range(30):
            x = base_x + float(self.np_random.uniform(-jx, jx))
            y = base_y + float(self.np_random.uniform(-jy, jy))
            theta = base_theta + float(self.np_random.uniform(-jt, jt))
            self.car.commit((x, y, theta, 0.0, 0.0))
            if not self.world.car_collides(self.car.obb()):
                return
        # fallback: đúng vị trí gốc
        self.car.commit((base_x, base_y, base_theta, 0.0, 0.0))

    def _pick_target_slot(self):
        if not self.random_target_slot:
            if 0 not in self.decoy_slots:
                return 0
            free = self.world.free_slot_indices()
            return int(free[0]) if free else 0
        free = self.world.free_slot_indices()
        if not free:
            free = list(range(self.num_slots))
        # "chỉ thay đổi target": ưu tiên slot khác episode trước
        if len(free) > 1:
            choices = [i for i in free if i != self.target_index]
            if choices:
                free = choices
        return int(free[int(self.np_random.integers(len(free)))])

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.episode_index += 1

        # Layout decoy chỉ đổi SAU khi có success (xem _on_success).
        if self._pending_decoy_spawn:
            self._spawn_decoy_layout()

        self._spawn_car()
        self.target_index = self._pick_target_slot()
        self.world.target_index = self.target_index

        slot = self.world.slots[self.target_index]
        self.prev_dist = math.hypot(slot.cx - self.car.x, slot.cy - self.car.y)
        self.step_count = 0

        info = {
            "target_slot": self.target_index,
            "episode_index": self.episode_index,
            "decoy_spawned": self.decoy_spawned,
            "layout_index": self.layout_index,
            "layout_successes": self.layout_successes,
            "total_successes": self.total_successes,
        }
        return self._get_obs(), info

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(self, action):
        throttle_dir, steer_dir = self._actions[int(action)]
        old = (self.car.x, self.car.y, self.car.theta, self.car.v, self.car.steer)

        # Chia nhỏ bước để tránh xuyên vật cản
        n_sub = 3
        sub_dt = self.dt / n_sub
        collided = False
        collided_obb = None
        for _ in range(n_sub):
            v = throttle_dir * self.car.max_speed_fwd * 0.5
            steer = steer_dir * self.car.max_steer
            L = self.car.wheelbase
            self.car.theta += (v / L) * math.tan(steer) * sub_dt
            self.car.x += v * math.cos(self.car.theta) * sub_dt
            self.car.y += v * math.sin(self.car.theta) * sub_dt
            self.car.v = v
            self.car.steer = steer
            if self.world.car_collides(self.car.obb()):
                collided = True
                collided_obb = self.car.obb()
                break

        if collided:
            self.car.commit(old)
            self.car.v = 0.0

        self.step_count += 1

        slot = self.world.slots[self.target_index]
        dist = math.hypot(slot.cx - self.car.x, slot.cy - self.car.y)
        dtheta = abs(
            ((self.car.theta - slot.theta + math.pi) % (2 * math.pi)) - math.pi
        )

        # Bảng phạt chi tiết theo từng loại object & tile cấm
        COLLISION_PENALTIES = {
            "decoy": -15.0,
            "lamp": -15.0,
            "boundary": -12.0,
            "bush": -10.0,
            "planting_tile": -10.0,
            "sidewalk": -8.0,
            "obstacle": -10.0,
        }

        get_tile = getattr(self.world, "get_tile_at", lambda x, y: None)
        current_tile = get_tile(self.car.x, self.car.y)
        if current_tile == "R":
            step_cost = -0.01       # Đường chính
        elif current_tile == "P":
            step_cost = -0.01 if dist < 28.0 else -0.15  # Đúng ô đích thì nhẹ, đè ô P khác thì phạt
        else:
            step_cost = -0.02       # Sân bãi 'C' hoặc map cũ

        reward = step_cost
        reward += 0.05 * (self.prev_dist - dist)
        self.prev_dist = dist

        terminated = False
        truncated = False
        info = {
            "target_slot": self.target_index,
            "episode_index": self.episode_index,
            "decoy_spawned": self.decoy_spawned,
            "layout_index": self.layout_index,
            "layout_successes": self.layout_successes,
            "total_successes": self.total_successes,
            "is_success": False,
            "is_collision": False,
            "collision_type": None,
        }

        if collided:
            check_detail = getattr(self.world, "check_collision_detail", lambda obb: None)
            c_type = check_detail(collided_obb) or "obstacle"
            reward += COLLISION_PENALTIES.get(c_type, -10.0)
            terminated = True
            info["is_collision"] = True
            info["collision_type"] = c_type

        if dist < self.success_dist and dtheta < self.success_theta and not collided:
            reward += 200.0
            terminated = True
            info["is_success"] = True
            self._on_success(info)

        if self.step_count >= self.max_steps:
            truncated = True

        return self._get_obs(), reward, terminated, truncated, info
