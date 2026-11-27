import math
import random

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from mapark.car import BicycleCar
from Unimportant.geometry import OBB, obb_intersect

_obb_hit = obb_intersect


def _normalize_angle(a):
    return ((a + math.pi) % (2 * math.pi)) - math.pi
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

        # --- Map: Æ°u tiÃªn spec dá»±ng tá»« 2 file JSON, fallback map JSON/config cÅ© ---
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
            raise ValueError("Map cáº§n Ã­t nháº¥t má»™t slot Ä‘á»—")

        # --- Decoy ---
        # `decoy_count`      : sá»‘ decoy má»—i layout decoy
        # `decoy_first_count`: sá»‘ decoy á»Ÿ láº§n init Äáº¦U (0 = bÃ£i trá»‘ng)
        # `decoy_on_reset`   : Ä‘áº·t decoy ngay tá»« episode 1 (thay vÃ¬ bÃ£i trá»‘ng)
        self.decoy_count = int(self.train_cfg.get("decoy_count", 0))
        self.decoy_seed = int(self.train_cfg.get("decoy_seed", 0))
        self.decoy_success_target = max(1, int(self.train_cfg.get("decoy_success_target", 20)))
        self.decoy_on_reset = bool(self.train_cfg.get("decoy_on_reset", False))
        self.decoy_first_count = int(
            self.train_cfg.get("decoy_first_count", self.decoy_count)
        )
        # decoy Ã©p ngay tá»« init thay vÃ¬ Ä‘á»ƒ bÃ£i trá»‘ng
        self.force_hard_start = bool(self.train_cfg.get("force_hard_start", False))
        if self.force_hard_start:
            self.decoy_first_count = self.decoy_count
        self.decoy_first_count = max(0, min(self.decoy_first_count, self.num_slots - 1))
        # luÃ´n chá»«a Ã­t nháº¥t 1 slot trá»‘ng lÃ m Ä‘Ã­ch
        self.decoy_count = max(0, min(self.decoy_count, self.num_slots - 1))
        self.easy_layout_done = False     # Ä‘Ã£ cháº¡y xong layout bÃ£i trá»‘ng

        self.success_dist = float(self.train_cfg.get("success_dist", 15.0))
        self.success_theta = math.radians(
            float(self.train_cfg.get("success_theta_deg", 20.0))
        )
        self.sensor_range = float(self.train_cfg.get("sensor_range", 180.0))
        self.obs_dist_scale = float(self.train_cfg.get("obs_dist_scale", 800.0))

        jitter = self.train_cfg.get("start_jitter", [30.0, 30.0, 0.3])
        self.start_jitter = (float(jitter[0]), float(jitter[1]), float(jitter[2]))

        # Spawn point / target chá»‰ Ä‘á»•i khi _layout_changed() == True.
        self.random_start_pose = bool(self.train_cfg.get("random_start_pose", True))

        # Map má»›i: Ä‘Ã­ch ngáº«u nhiÃªn trong cÃ¡c slot trá»‘ng.
        # Config kiá»ƒu cÅ©: giá»¯ slot 0 Ä‘á»ƒ khÃ´ng phÃ¡ model Ä‘Ã£ train trÆ°á»›c Ä‘Ã¢y.
        default_random_target = "map" in config
        self.random_target_slot = bool(
            self.train_cfg.get("random_target_slot", default_random_target)
        )
        self._locked_target_index = 0
        self._locked_car_pose = None
        self.layout_target_index = None   # target Ä‘Ã£ chá»‘t cho layout decoy hiá»‡n táº¡i

        # --- Curriculum: 3 má»‘c Ä‘á»•i khÃ¡c nhau ---
        # 20 success: Ä‘á»•i nháº¹ spawn point (trÆ°á»£t Â±spawn_shift_px, trÃªn road)
        # 60 success: Ä‘á»•i chá»— Ä‘á»— (target, cÃ¹ng hÆ°á»›ng)
        # 180 success: reset toÃ n bá»™ (decoy + spawn + target + 3 bá»™ Ä‘áº¿m)
        self.spawn_change_every = max(1, int(self.train_cfg.get("spawn_change_every", 20)))
        self.target_change_every = max(1, int(self.train_cfg.get("target_change_every", 60)))
        self.decoy_change_every = max(1, int(self.train_cfg.get("decoy_change_every", 180)))
        self.spawn_shift_px = float(self.train_cfg.get("spawn_shift_px", 30.0))
        self.max_episodes_per_stage = max(
            0, int(self.train_cfg.get("max_episodes_per_stage", 300))
        )
        self.spawn_successes = 0
        self.target_successes = 0
        self.decoy_successes = 0
        self.spawn_level = 0          # sá»‘ má»‘c spawn Ä‘Ã£ qua trong chu ká»³ 180
        self.spawn_base_pose = None   # gá»‘c spawn cá»§a layout (trÆ°á»£t quanh gá»‘c nÃ y)
        self.stage_start_episode = 0
        self._stage_log = []

        # --- Phase 1 "easy": bá» bá»›t logic 3D trong `easy_successes` success Ä‘áº§u ---
        # TrÆ°á»›c 180 success: chá»‰ cáº§n tÃ¢m xe gáº§n slot + hÆ°á»›ng gáº§n Ä‘Ãºng, khÃ´ng cáº§n
        # 4 gÃ³c trong slot / Ä‘á»©ng yÃªn / hold 10 bÆ°á»›c. Sau Ä‘Ã³ quay vá» Ä‘Ãºng logic
        # ParkingAgent 3D (giá»¯ nguyÃªn nhÆ° HEAD).
        self.curriculum_easy = bool(self.train_cfg.get("curriculum_easy", True))
        self.easy_successes = max(1, int(self.train_cfg.get("easy_successes", 180)))
        self.easy_success_dist = float(self.train_cfg.get("easy_success_dist", 30.0))
        self.easy_success_theta_deg = float(
            self.train_cfg.get("easy_success_theta_deg", 45.0)
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
            low=-1.0, high=1.0, shape=(18,), dtype=np.float32
        )
        # Giá»‘ng ParkingAgent 3D: 3 action liÃªn tá»¥c [ga/lÃ¹i, lÃ¡i, phanh] trong [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

        # TÆ°Æ¡ng thÃ­ch ngÆ°á»£c: ká»‹ch báº£n/test cÅ© truyá»n sá»‘ nguyÃªn -> dÃ¹ng báº£ng
        # 6 action rá»i ráº¡c (chá»‰ Ä‘á»ƒ test, KHÃ”NG dÃ¹ng khi train).
        self._actions = [
            (1, -1), (1, 0), (1, 1),
            (-1, -1), (-1, 0), (-1, 1),
        ]

        self.brake_decel = float(self.train_cfg.get("brake_decel", 200.0))
        self.use_kinematic = bool(self.train_cfg.get("use_kinematic", True))

        self.step_penalty = float(self.train_cfg.get("step_penalty", -0.05))
        self.progress_weight = float(self.train_cfg.get("progress_weight", 0.02))
        self.success_reward = float(self.train_cfg.get("success_reward", 150.0))

        # --- Reward khi va cháº¡m, riÃªng cho tá»«ng loáº¡i tile / object ---
        self.collision_reward = float(self.train_cfg.get("collision_reward", -10.0))
        default_kind = {
            "wall": self.collision_reward,
            "tile:G": -10.0,        # planting
            "tile:H": -10.0,        # sidewalk
            "object:B": -6.0,       # bush
            "object:L": -6.0,       # lamp
            "decoy": -10.0,
            "obstacle": self.collision_reward,
            "boundary": self.collision_reward,
        }
        default_kind.update(
            {str(k): float(v) for k, v in self.train_cfg.get("collision_rewards", {}).items()}
        )
        self.collision_rewards = default_kind
        self.terminate_on_collision = bool(
            self.train_cfg.get("terminate_on_collision", True)
        )

        # --- Äá»“ng bá»™ vá»›i ParkingAgent (Unity 3D) ---
        self.max_angle_deg = float(self.train_cfg.get("suite_max_angle_deg", 10.0))
        self.max_speed_to_park = float(self.train_cfg.get("suite_max_speed_to_park", 0.3))
        self.hold_decisions = max(1, int(self.train_cfg.get("suite_hold_decisions", 10)))
        self.obs_scale = float(self.train_cfg.get("suite_obs_scale", 15.0))
        self.max_distance = float(self.train_cfg.get("suite_max_distance", 40.0))
        self.distance_shaping = float(self.train_cfg.get("suite_distance_shaping", 0.05))
        self.align_shaping = float(self.train_cfg.get("suite_align_shaping", 0.02))
        self.use_suite_parked = bool(self.train_cfg.get("suite_parked_check", True))
        self.parked_count = 0
        self.prev_align = 0.0
        self.start_episode_dist = 0.0

    # ------------------------------------------------------------------
    # Curriculum phase 1 (easy)
    # ------------------------------------------------------------------
    @property
    def in_easy_phase(self):
        """True trong `easy_successes` success Ä‘áº§u: cÃ²n á»Ÿ logic rÃºt gá»n."""
        if not self.curriculum_easy:
            return False
        return self.total_successes < self.easy_successes

    # ------------------------------------------------------------------
    # Dá»±ng world
    # ------------------------------------------------------------------
    def _build_world(self):
        """World kiá»ƒu cÅ© (config YAML cÃ³ world/car/obstacles/slots)."""
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
        """Äáº·t `decoy_count` xe vÃ o slot ngáº«u nhiÃªn (layout má»›i).

        Layout táº¥t Ä‘á»‹nh theo `decoy_seed + layout_index` Ä‘á»ƒ tÃ¡i láº­p Ä‘Æ°á»£c.
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
            # xe decoy Ä‘á»— Ä‘Ãºng tÃ¢m slot, cÃ¹ng hÆ°á»›ng vá»›i slot
            decoys.append(
                OBB(slot.cx, slot.cy, self.car.length, self.car.width, slot.theta)
            )
        self.world.set_decoys(decoys)
        self.world.decoy_slots = ids
        self.decoy_slots = ids

    def force_decoy_layout(self):
        """Äá»•i layout ngay á»Ÿ láº§n reset káº¿ tiáº¿p (dÃ¹ng cho demo/eval)."""
        self._pending_decoy_spawn = True

    def _on_success(self, info):
        """Cá»™ng success vÃ  xÃ©t 3 má»‘c curriculum.

        - `spawn_change_every` (20): Ä‘á»•i nháº¹ spawn point
        - `target_change_every` (60): Ä‘á»•i chá»— Ä‘á»— (cÃ¹ng hÆ°á»›ng)
        - `decoy_change_every` (180): reset toÃ n bá»™ + reset cáº£ 3 bá»™ Ä‘áº¿m
        """
        self.total_successes += 1
        self.layout_successes += 1
        self.spawn_successes += 1
        self.target_successes += 1
        self.decoy_successes += 1

        # 180: reset toÃ n bá»™ (decoy + spawn + target + 3 bá»™ Ä‘áº¿m)
        if self.decoy_successes >= self.decoy_change_every:
            info["decoy_event"] = "respawn"
            info["completed_successes"] = self.decoy_successes
            self._pending_decoy_spawn = True
            self.spawn_successes = 0
            self.target_successes = 0
            self.decoy_successes = 0
            self.spawn_level = 0
            if self.max_episodes_per_stage:
                self.stage_start_episode = self.episode_index
            info["stage_events"] = ["decoy"]

        info["layout_index"] = self.layout_index
        info["layout_successes"] = self.layout_successes
        info["total_successes"] = self.total_successes
        info["decoy_spawned"] = self.decoy_spawned
        info["spawn_successes"] = self.spawn_successes
        info["target_successes"] = self.target_successes
        info["decoy_successes"] = self.decoy_successes
        info["spawn_level"] = self.spawn_level
        info["spawn_shift_px"] = self.spawn_shift_px

    # ------------------------------------------------------------------
    # Quan sÃ¡t / reset
    # ------------------------------------------------------------------
    def _sensors(self):
        return sense_all(self.world, self.car, max_range=self.sensor_range)

    def _slot_axes(self, slot):
        """Trá»¥c dÃ i / trá»¥c ngáº¯n cá»§a slot.

        OBB gÃ³c `theta`: cáº¡nh `w` náº±m dá»c trá»¥c (cos theta, sin theta).
        Náº¿u w < h thÃ¬ cáº¡nh dÃ i thá»±c cháº¥t lÃ  trá»¥c phÃ¡p tuyáº¿n (nhÆ° slot 44x56).
        """
        ex = (math.cos(slot.theta), math.sin(slot.theta))
        ey = (-math.sin(slot.theta), math.cos(slot.theta))
        if slot.w >= slot.h:
            return ex, ey
        return ey, ex

    def _angle_to_axis(self, slot):
        """Lá»‡ch giá»¯a mÅ©i xe vÃ  trá»¥c dÃ i slot, -90..90 Ä‘á»™ (tiáº¿n/lÃ¹i nhÆ° nhau)."""
        long_axis, _ = self._slot_axes(slot)
        fx, fy = math.cos(self.car.theta), math.sin(self.car.theta)
        ang = math.degrees(
            math.atan2(fx * long_axis[1] - fy * long_axis[0],
                       fx * long_axis[0] + fy * long_axis[1])
        )
        if ang > 90.0:
            ang -= 180.0
        elif ang < -90.0:
            ang += 180.0
        return ang

    def _slot_contains(self, obb, point):
        """Äiá»ƒm cÃ³ náº±m trong OBB slot khÃ´ng."""
        c, s = math.cos(-obb.theta), math.sin(-obb.theta)
        dx, dy = point[0] - obb.cx, point[1] - obb.cy
        lx = c * dx - s * dy
        ly = s * dx + c * dy
        return abs(lx) <= obb.w / 2.0 and abs(ly) <= obb.h / 2.0

    def _is_parked(self, slot):
        """Äiá»u kiá»‡n Ä‘á»— giá»‘ng ParkingAgent: gÃ³c, váº­n tá»‘c, cáº£ 4 gÃ³c xe trong slot."""
        if abs(self._angle_to_axis(slot)) > self.max_angle_deg:
            return False
        if abs(self.car.v) > self.max_speed_to_park:
            return False
        return all(self._slot_contains(slot, p) for p in self.car.obb().corners())

    # ------------------------------------------------------------------
    # Test helper: dá»±ng tráº¡ng thÃ¡i "Ä‘á»— chuáº©n" (dÃ¹ng trong test, khÃ´ng áº£nh hÆ°á»Ÿng train)
    # ------------------------------------------------------------------
    def place_parked(self):
        """Äáº·t xe Ä‘Ãºng tÃ¢m slot Ä‘Ã­ch, Ä‘Ãºng hÆ°á»›ng, Ä‘á»©ng yÃªn."""
        slot = self.world.slots[self.target_index]
        long_axis, _ = self._slot_axes(slot)
        theta = math.atan2(long_axis[1], long_axis[0])
        self.car.commit((slot.cx, slot.cy, theta, 0.0, 0.0))
        self.prev_dist = 0.0
        self.prev_align = 1.0
        self.parked_count = 0
        return slot

    def _get_obs(self):
        car = self.car
        sensors = self._sensors()

        sensor_vals = []
        for name in self.SENSOR_ORDER:
            dist = sensors[name][0]
            # Gáº§n váº­t cáº£n â†’ gáº§n 1; trá»‘ng â†’ gáº§n 0
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

        # --- Obs bá»• sung Ä‘á»ƒ bÃ¡m sÃ¡t 3D ---
        c_s, s_s = math.cos(-slot.theta), math.sin(-slot.theta)
        local_x = c_s * dx - s_s * dy      # vá»‹ trÃ­ xe trong há»‡ slot
        local_y = s_s * dx + c_s * dy

        # hÆ°á»›ng mÅ©i xe trong há»‡ slot: (cos, sin) cá»§a (theta - slot.theta)
        theta_rel = car.theta - slot.theta
        fwd_slot_x = math.cos(theta_rel)
        fwd_slot_y = math.sin(theta_rel)

        # tÃ¢m slot trong há»‡ xe (láº¥y tá»« slot ngÆ°á»£c sang xe)
        to_slot_x = c * dx - s * dy
        to_slot_y = s * dx + c * dy

        obs = sensor_vals + [
            v_norm,
            steer_norm,
            math.cos(ang),
            math.sin(ang),
            min(1.0, dist / self.obs_dist_scale),
            # 3D parity (5 chiá»u)
            local_x / self.obs_scale,
            local_y / self.obs_scale,
            fwd_slot_x,
            fwd_slot_y,
            self._angle_to_axis(slot) / 90.0,
        ]
        return np.array(obs, dtype=np.float32)

    # ------------------------------------------------------------------
    # Spawn point
    # ------------------------------------------------------------------
    def _world_cell(self, x, y):
        """(row, col) cá»§a Ã´ chá»©a Ä‘iá»ƒm, None náº¿u ngoÃ i map."""
        ts = getattr(self.world, "cell_size", None)
        if not ts:
            return None
        r = int(y // ts)
        c = int(x // ts)
        return r, c

    def _cell_is_road(self, row, col):
        """Ã” cÃ³ pháº£i Ä‘Æ°á»ng (tile 'R') vÃ  khÃ´ng bá»‹ decoy chiáº¿m."""
        if row is None or col is None:
            return False
        layout = getattr(self.world, "tiles_raw", None) or []
        if not (0 <= row < len(layout)) or not (0 <= col < len(layout[row])):
            return False
        if layout[row][col] != "R":
            return False
        ts = self.world.cell_size
        cx = col * ts + ts / 2.0
        cy = row * ts + ts / 2.0
        probe = OBB(cx, cy, self.car.length, self.car.width, self.start_pose[2])
        return not any(
            _obb_hit(probe, d) for d in self.world.decoys
        )

    def _draw_car_pose(self):
        """Random 1 pose xuáº¥t phÃ¡t quanh gá»‘c, váº«n náº±m trÃªn Ä‘Æ°á»ng (tile R).

        Má»©c trÆ°á»£t tÄƒng dáº§n theo sá»‘ má»‘c Ä‘Ã£ qua trong chu ká»³ 180:
          0 má»‘c  -> jitter nhá» quanh gá»‘c
          >=1    -> trÆ°á»£t ngáº«u nhiÃªn trong Â±spawn_shift_px quanh gá»‘c
        """
        base_x, base_y, base_theta = (
            self.spawn_base_pose or self.start_pose
        )
        jx, jy, jt = self.start_jitter
        shift = self.spawn_shift_px
        shifted = self.spawn_level > 0

        for i in range(40):
            if shifted:
                # trÆ°á»£t Â±spawn_shift_px quanh vá»‹ trÃ­ spawn gá»‘c cá»§a layout
                x = base_x + float(self.np_random.uniform(-shift, shift))
                y = base_y + float(self.np_random.uniform(-shift, shift))
            else:
                x = base_x + float(self.np_random.uniform(-jx, jx))
                y = base_y + float(self.np_random.uniform(-jy, jy))
            theta = base_theta + float(self.np_random.uniform(-jt, jt))

            cell = self._world_cell(x, y)
            if cell is None or not self._cell_is_road(*cell):
                continue

            self.car.commit((x, y, theta, 0.0, 0.0))
            if not self.world.car_collides(self.car.obb()):
                return (x, y, theta)
        # fallback: Ä‘Ãºng vá»‹ trÃ­ gá»‘c
        self.car.commit((base_x, base_y, base_theta, 0.0, 0.0))
        return (base_x, base_y, base_theta)

    def _spawn_car(self):
        """Äáº·t xe vÃ o pose Ä‘Ã£ khoÃ¡ cá»§a stage; chá»‰ random láº¡i khi cÃ³ reset"""
        if self._locked_car_pose is None:
            self._locked_car_pose = self._draw_car_pose()
        x, y, theta = self._locked_car_pose
        self.car.commit((x, y, theta, 0.0, 0.0))

    def _pick_target_slot(self):
        if not self.random_target_slot:
            if 0 not in self.decoy_slots:
                return 0
            free = self.world.free_slot_indices()
            return int(free[0]) if free else 0
        free = self.world.free_slot_indices()
        if not free:
            free = list(range(self.num_slots))

        base = (
            self.layout_target_index
            if self.layout_target_index is not None
            else self._locked_target_index
        )
        # Má»‘c 60: Æ°u tiÃªn slot khÃ¡c target gá»‘c cá»§a layout -> tháº¥y khÃ¡c biá»‡t rÃµ
        if len(free) > 1 and base is not None:
            choices = [i for i in free if i != base]
            if choices:
                free = choices
        return int(free[int(self.np_random.integers(len(free)))])

    def _lock_target_slot(self):
        """Chá»‘t target cho layout hiá»‡n táº¡i; chá»‰ Ä‘á»•i khi tá»›i má»‘c 60 hoáº·c 180."""
        if (
            self._locked_target_index is None
            or self._locked_target_index in self.decoy_slots
        ):
            self._locked_target_index = self._pick_target_slot()
        self.layout_target_index = self._locked_target_index
        return self._locked_target_index

    def _reset_layout(self):
        """Random láº¡i toÃ n bá»™ layout (decoy + spawn point + target).

        Láº§n init Ä‘áº§u tiÃªn dÃ¹ng `decoy_first_count` (máº·c Ä‘á»‹nh 0 = bÃ£i trá»‘ng),
        cÃ¡c láº§n sau dÃ¹ng `decoy_count`.
        """
        self._locked_car_pose = None
        self._locked_target_index = None
        self.layout_target_index = None

        self.layout_successes = 0
        self._pending_decoy_spawn = False
        self.spawn_level = 0
        self.spawn_base_pose = None

        first_init = not self.easy_layout_done
        n_decoy = self.decoy_first_count if first_init else self.decoy_count
        if first_init:
            self.easy_layout_done = True

        if n_decoy <= 0 or self.num_slots <= 1:
            self.world.set_decoys([])
            self.world.decoy_slots = []
            self.decoy_slots = []
            self.decoy_spawned = False
            self.first_layout_index = self.layout_index
            return

        self.layout_index += 1
        rng = random.Random(self.decoy_seed * 1_000_003 + self.layout_index)
        k = min(n_decoy, self.num_slots - 1)
        ids = sorted(rng.sample(range(self.num_slots), k))

        decoys = []
        for i in ids:
            slot = self.world.slots[i]
            # xe decoy Ä‘á»— Ä‘Ãºng tÃ¢m slot, cÃ¹ng hÆ°á»›ng vá»›i slot
            decoys.append(
                OBB(slot.cx, slot.cy, self.car.length, self.car.width, slot.theta)
            )
        self.world.set_decoys(decoys)
        self.world.decoy_slots = ids
        self.decoy_slots = ids
        self.decoy_spawned = True

    def _advance_stages(self, last_step=False):
        """Äá»•i stage. Gá»i tá»« reset(), hoáº·c khi má»™t stage kÃ©o quÃ¡ lÃ¢u.

        Tráº£ list sá»± kiá»‡n: "spawn" | "target" | "decoy".
        """
        events = []
        if self._pending_decoy_spawn:
            events.append("decoy")
            self._pending_decoy_spawn = False
            self.spawn_successes = 0
            self.target_successes = 0
            self.decoy_successes = 0
            self.spawn_level = 0
            self.stage_start_episode = self.episode_index
            return events

        # Target: Ä‘á»•i theo má»‘c tuyá»‡t Ä‘á»‘i trong chu ká»³ 180
        if self.target_successes >= self.target_change_every:
            self.target_successes = 0
            self._locked_target_index = None
            self.layout_target_index = None
            self.stage_start_episode = self.episode_index
            events.append("target")

        # Spawn: má»—i `spawn_change_every` success
        if self.spawn_successes >= self.spawn_change_every:
            self.spawn_successes = 0
            self.spawn_level += 1
            self._locked_car_pose = None
            self.stage_start_episode = self.episode_index
            events.append("spawn")

        return events

    def _force_stage_advance(self):
        """Káº¹t quÃ¡ `max_episodes_per_stage` -> buá»™c tiáº¿n stage gáº§n nháº¥t."""
        if not self.max_episodes_per_stage:
            return []
        if self.episode_index - self.stage_start_episode < self.max_episodes_per_stage:
            return []

        # Chá»n má»‘c chÆ°a Ä‘áº¡t cÃ³ ngÆ°á»¡ng nhá» nháº¥t
        pending = []
        if self.spawn_successes < self.spawn_change_every:
            pending.append(("spawn", self.spawn_successes, self.spawn_change_every))
        if self.target_successes < self.target_change_every:
            pending.append(("target", self.target_successes, self.target_change_every))
        if not pending:
            return []

        kind, _, _ = min(pending, key=lambda x: x[1])
        self.stage_start_episode = self.episode_index
        if kind == "spawn":
            self.spawn_successes = 0
            self.spawn_level += 1
            self._locked_car_pose = None
        else:
            self.target_successes = 0
            self._locked_target_index = None
        return [kind]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.episode_index += 1

        # Buá»™c Ä‘á»•i stage náº¿u agent káº¹t quÃ¡ lÃ¢u á»Ÿ stage hiá»‡n táº¡i
        forced = self._force_stage_advance()

        if self._pending_decoy_spawn or (self.decoy_on_reset and self.layout_index == 0):
            self._reset_layout()

        stage_events = self._advance_stages() + forced

        self._spawn_car()
        self.target_index = self._lock_target_slot()
        self.world.target_index = self.target_index

        slot = self.world.slots[self.target_index]
        self.prev_dist = math.hypot(slot.cx - self.car.x, slot.cy - self.car.y)
        self.start_episode_dist = self.prev_dist
        self.prev_align = 1.0 - abs(self._angle_to_axis(slot)) / 90.0
        self.parked_count = 0
        self.step_count = 0

        info = {
            "target_slot": self.target_index,
            "episode_index": self.episode_index,
            "decoy_spawned": self.decoy_spawned,
            "layout_index": self.layout_index,
            "layout_successes": self.layout_successes,
            "total_successes": self.total_successes,
            "spawn_successes": self.spawn_successes,
            "target_successes": self.target_successes,
            "decoy_successes": self.decoy_successes,
            "spawn_level": self.spawn_level,
            "spawn_shift_px": self.spawn_shift_px,
            "stage_events": stage_events,
            "forced_stage": list(forced),
        }
        return self._get_obs(), info

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def _parse_action(self, action):
        """Chuáº©n hoÃ¡ action vá» (throttle, steer, brake) trong [-1, 1].

        - Box(3,):  [ga/lÃ¹i, lÃ¡i, phanh] nhÆ° ParkingAgent 3D.
        - Sá»‘ nguyÃªn: váº«n cháº¥p nháº­n báº£ng 6 action cÅ© (chá»‰ Ä‘á»ƒ test/eval).
        """
        if np.isscalar(action) or np.ndim(action) == 0:
            throttle_dir, steer_dir = self._actions[int(action)]
            return float(throttle_dir), float(steer_dir), -1.0

        a = np.asarray(action, dtype=np.float64).reshape(-1)
        if a.size < 3:
            a = np.concatenate([a, np.full(3 - a.size, -1.0)])
        throttle = float(np.clip(a[0], -1.0, 1.0))
        steer = float(np.clip(a[1], -1.0, 1.0))
        brake = float(np.clip(a[2], -1.0, 1.0))
        return throttle, steer, brake

    def _kinematic_step(self, throttle, steer, brake, dt):
        """Má»™t bÆ°á»›c Ä‘á»™ng há»c kiá»ƒu CarController: gia tá»‘c + chá»‘ng Ä‘áº£o chiá»u.

        Tráº£ list tráº¡ng thÃ¡i con (Ä‘Ã£ commit vÃ o self.car).
        """
        car = self.car
        n_sub = 3
        sub_dt = dt / n_sub
        states = []
        for _ in range(n_sub):
            if sub_dt <= 0:
                car.v = 0.0
                break

            # --- LÃ¡i: tá»‘c Ä‘á»™ Ä‘Ã¡nh lÃ¡i giá»›i háº¡n (steer_rate) ---
            target_steer = steer * car.max_steer
            diff = target_steer - car.steer
            step = car.steer_rate * sub_dt
            new_steer = car.steer + max(-step, min(step, diff))

            # --- Ga / phanh / trÃ´i ---
            new_v = car.v
            engine = throttle > 0.0
            braking = brake > 0.0 and new_v != 0.0
            if braking:
                sign = 1.0 if new_v > 0 else -1.0
                new_v -= sign * self.brake_decel * sub_dt
                if (new_v > 0) != (car.v > 0):
                    new_v = 0.0
            elif abs(throttle) > 1e-6:
                if engine:
                    new_v += car.accel * abs(throttle) * sub_dt
                else:
                    # LÃ¹i: pháº£i phanh vá» 0 trÆ°á»›c, chÆ°a Ä‘áº£o chiá»u ngay
                    if car.v > 0.0:
                        new_v -= self.brake_decel * sub_dt
                        if new_v < 0.0:
                            new_v = 0.0
                    else:
                        new_v -= car.accel * abs(throttle) * sub_dt
            else:
                # Nháº£ ga: xe trÃ´i, giáº£m dáº§n theo decel
                if new_v > 0.0:
                    new_v = max(0.0, new_v - car.decel * sub_dt)
                elif new_v < 0.0:
                    new_v = min(0.0, new_v + car.decel * sub_dt)
            new_v = max(-car.max_speed_rev, min(car.max_speed_fwd, new_v))

            # --- Bicycle model ---
            new_theta = car.theta + (new_v / car.wheelbase) * math.tan(new_steer) * sub_dt
            new_x = car.x + new_v * math.cos(new_theta) * sub_dt
            new_y = car.y + new_v * math.sin(new_theta) * sub_dt

            car.commit((new_x, new_y, new_theta, new_v, new_steer))
            states.append((new_x, new_y, new_theta, new_v, new_steer))
        return states

    def _direct_step(self, throttle_dir, steer_dir, dt):
        """Äá»™ng há»c cÅ©: ga/lÃ¡i tá»©c thá»i (giá»¯ cho tÆ°Æ¡ng thÃ­ch ngÆ°á»£c)."""
        car = self.car
        n_sub = 3
        sub_dt = dt / n_sub
        states = []
        for _ in range(n_sub):
            v = throttle_dir * car.max_speed_fwd * 0.5
            steer = steer_dir * car.max_steer
            L = car.wheelbase
            car.theta += (v / L) * math.tan(steer) * sub_dt
            car.x += v * math.cos(car.theta) * sub_dt
            car.y += v * math.sin(car.theta) * sub_dt
            car.v = v if sub_dt > 0 else 0.0
            car.steer = steer
            states.append((car.x, car.y, car.theta, car.v, car.steer))
        return states

    def step(self, action, dt=None):
        throttle, steer, brake = self._parse_action(action)
        old = (self.car.x, self.car.y, self.car.theta, self.car.v, self.car.steer)
        dt = self.dt if dt is None else dt

        is_discrete = np.isscalar(action) or np.ndim(action) == 0
        use_kinematic = self.use_kinematic and not is_discrete

        hits = []
        if use_kinematic:
            for state in self._kinematic_step(throttle, steer, brake, dt):
                self.car.commit(state)
                hits = self.world.collisions(self.car.obb())
                if hits:
                    break
        elif is_discrete:
            throttle_dir, steer_dir = self._actions[int(action)]
            for state in self._direct_step(throttle_dir, steer_dir, dt):
                self.car.commit(state)
                hits = self.world.collisions(self.car.obb())
                if hits:
                    break
        else:
            for state in self._direct_step(throttle, steer, dt):
                self.car.commit(state)
                hits = self.world.collisions(self.car.obb())
                if hits:
                    break

        collided = bool(hits)
        if collided:
            self.car.commit(old)
            self.car.v = 0.0

        self.step_count += 1

        slot = self.world.slots[self.target_index]
        dist = math.hypot(slot.cx - self.car.x, slot.cy - self.car.y)
        align = 1.0 - abs(self._angle_to_axis(slot)) / 90.0

        # Va cháº¡m: bá» thÆ°á»Ÿng progress Ä‘á»ƒ agent khÃ´ng Ä‘Æ°á»£c "thÆ°á»Ÿng" khi lao vÃ o
        # váº­t cáº£n á»Ÿ gáº§n Ä‘Ã­ch.
        if collided:
            progress = 0.0
        else:
            progress = self.progress_weight * (self.prev_dist - dist)
            progress += self.align_shaping * (align - self.prev_align)
        self.prev_dist = dist
        self.prev_align = align

        dtheta = abs(
            ((self.car.theta - slot.theta + math.pi) % (2 * math.pi)) - math.pi
        )

        # Reward shaping
        reward = self.step_penalty + progress

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
            "collision_kinds": [],
            "collision_reward": 0.0,
            "align": align,
            "parked_count": self.parked_count,
        }

        if collided:
            kinds = sorted({h["kind"] for h in hits})
            penalties = [self.collision_rewards.get(k, self.collision_reward) for k in kinds]
            # Cháº¡m nhiá»u loáº¡i cÃ¹ng lÃºc: dÃ¹ng má»©c pháº¡t náº·ng nháº¥t, khÃ´ng cá»™ng dá»“n
            total = min(penalties)
            reward += total
            info["is_collision"] = True
            info["collision_kinds"] = kinds
            info["collision_reward"] = total
            if self.terminate_on_collision:
                terminated = True

        # 3D parity: Ä‘i quÃ¡ xa slot (maxDistance) -> fail. 0 = táº¯t.
        # Phase 1 ("easy") chÆ°a Ã¡p dá»¥ng Ä‘á»ƒ agent táº­p trung há»c Ä‘Æ°á»ng Ä‘i.
        if (
            not terminated
            and not self.in_easy_phase
            and self.max_distance > 0
            and dist > self.max_distance
        ):
            reward += self.collision_reward
            terminated = True
            info["is_collision"] = True
            info["collision_kinds"] = ["far_from_slot"]

        if not terminated:
            if self.in_easy_phase:
                # --- Phase 1 (easy): logic rut gon, khong can hold ---
                parked = (
                    dist < self.easy_success_dist
                    and abs(self._angle_to_axis(slot)) < self.easy_success_theta_deg
                )
                info['is_parked'] = bool(parked)
                if parked:
                    reward += self.success_reward
                    terminated = True
                    info['is_success'] = True
                    self._on_success(info)
                info['parked_count'] = 0
            else:
                # --- Phase 2: logic 3D day du (giu nguyen HEAD) ---
                if self.use_suite_parked:
                    parked = self._is_parked(slot)
                else:
                    parked = dist < self.success_dist and dtheta < self.success_theta

                info['is_parked'] = bool(parked)
                if parked:
                    self.parked_count += 1
                    if self.parked_count >= self.hold_decisions:
                        reward += self.success_reward
                        terminated = True
                        info['is_success'] = True
                        self._on_success(info)
                else:
                    self.parked_count = 0
                info['parked_count'] = self.parked_count

        info['easy_phase'] = self.in_easy_phase

        if self.step_count >= self.max_steps:
            truncated = True

        return self._get_obs(), reward, terminated, truncated, info
