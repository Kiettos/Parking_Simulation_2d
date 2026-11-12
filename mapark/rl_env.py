import math
import random

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from mapark.car import BicycleCar
from mapark.geometry import OBB, obb_intersect
from mapark.sensors import sense_all
from mapark.world import World


class ParkingEnv(gym.Env):
    metadata = {"render_modes": []}

    SENSOR_ORDER = ["F", "FR", "R", "BR", "B", "BL", "L", "FL"]

    def __init__(self, config, max_steps=400, dt=0.15):
        super().__init__()
        self.config = config
        self.max_steps = max_steps
        self.dt = dt
        self.world = self._build_world()
        self.car = self.world.car

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

    def _build_world(self):
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

    def _sensors(self):
        return sense_all(self.world, self.car, max_range=180.0)

    def _get_obs(self):
        car = self.car
        sensors = self._sensors()

        sensor_vals = []
        for name in self.SENSOR_ORDER:
            dist = sensors[name][0]
            # Gần vật cản → gần 1; trống → gần 0
            sensor_vals.append(1.0 - min(dist, 180.0) / 180.0)

        v_norm = max(-1.0, min(1.0, car.v / car.max_speed_fwd))
        steer_norm = max(-1.0, min(1.0, car.steer / car.max_steer))

        slot = self.world.slots[0]
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
            min(1.0, dist / 800.0),
        ]
        return np.array(obs, dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        c = self.config["car"]
        # Randomize start nhẹ để agent generalize
        self.car.x = c["start"][0] + random.uniform(-40, 40)
        self.car.y = c["start"][1] + random.uniform(-40, 40)
        self.car.theta = c["start"][2] + random.uniform(-0.4, 0.4)
        self.car.v = 0.0
        self.car.steer = 0.0
        self.step_count = 0

        slot = self.world.slots[0]
        self.prev_dist = math.hypot(slot.cx - self.car.x, slot.cy - self.car.y)
        return self._get_obs(), {}

    def _kinematic_step(self, throttle_dir, steer_dir):
        v = throttle_dir * self.car.max_speed_fwd * 0.5  # half speed
        steer = steer_dir * self.car.max_steer
        L = self.car.wheelbase
        dt = self.dt

        new_theta = self.car.theta + (v / L) * math.tan(steer) * dt
        new_x = self.car.x + v * math.cos(new_theta) * dt
        new_y = self.car.y + v * math.sin(new_theta) * dt
        return new_x, new_y, new_theta, v, steer

    def step(self, action):
        throttle_dir, steer_dir = self._actions[int(action)]
        old = (self.car.x, self.car.y, self.car.theta, self.car.v, self.car.steer)

        # Chia nhỏ bước để tránh xuyên vật cản
        n_sub = 3
        sub_dt = self.dt / n_sub
        collided = False
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
                break

        if collided:
            self.car.commit(old)
            self.car.v = 0.0

        self.step_count += 1

        slot = self.world.slots[0]
        dist = math.hypot(slot.cx - self.car.x, slot.cy - self.car.y)
        dtheta = abs(
            ((self.car.theta - slot.theta + math.pi) % (2 * math.pi)) - math.pi
        )

        # Reward shaping
        reward = -0.02                                    # step penalty
        reward += 0.05 * (self.prev_dist - dist)          # progress
        self.prev_dist = dist

        terminated = False
        truncated = False

        if collided:
            reward -= 10.0
            terminated = True

        if dist < 15.0 and dtheta < math.radians(20):
            reward += 200.0
            terminated = True

        if self.step_count >= self.max_steps:
            truncated = True

        return self._get_obs(), reward, terminated, truncated, {}