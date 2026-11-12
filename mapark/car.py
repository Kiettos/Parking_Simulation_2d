import math

from mapark.geometry import OBB


class BicycleCar:
    def __init__(self, x, y, theta, length=44, width=20, wheelbase=28):
        self.x = float(x)
        self.y = float(y)
        self.theta = float(theta)
        self.length = float(length)
        self.width = float(width)
        self.wheelbase = float(wheelbase)
        self.v = 0.0
        self.steer = 0.0

        self.max_speed_fwd = 160.0
        self.max_speed_rev = 80.0
        self.accel = 260.0
        self.decel = 220.0
        self.max_steer = math.radians(35)
        self.steer_rate = math.radians(150)

    def obb(self):
        return OBB(self.x, self.y, self.length, self.width, self.theta)

    def propose(self, dt, throttle, steer_input):
        """Tính trạng thái kế tiếp, KHÔNG gán vào self."""
        # Đánh lái
        target = steer_input * self.max_steer
        if steer_input == 0:
            new_steer = self.steer * max(0.0, 1.0 - 5.0 * dt)
        else:
            diff = target - self.steer
            step = self.steer_rate * dt
            new_steer = self.steer + max(-step, min(step, diff))

        # Ga / phanh
        new_v = self.v
        if throttle > 0:
            new_v += self.accel * dt
        elif throttle < 0:
            new_v -= self.accel * dt
        else:
            if new_v > 0:
                new_v = max(0.0, new_v - self.decel * dt)
            else:
                new_v = min(0.0, new_v + self.decel * dt)
        new_v = max(-self.max_speed_rev, min(self.max_speed_fwd, new_v))

        # Bicycle model
        new_theta = self.theta + (new_v / self.wheelbase) * math.tan(new_steer) * dt
        new_x = self.x + new_v * math.cos(new_theta) * dt
        new_y = self.y + new_v * math.sin(new_theta) * dt
        return new_x, new_y, new_theta, new_v, new_steer

    def commit(self, state):
        self.x, self.y, self.theta, self.v, self.steer = state