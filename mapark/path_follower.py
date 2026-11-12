import math


class PathFollower:
    def __init__(self, lookahead=45.0):
        self.path = []
        self.idx = 0
        self.lookahead = lookahead
        self.done = False

    def set_path(self, path):
        self.path = path
        self.idx = 0
        self.done = False

    def _target(self, car):
        if self.idx >= len(self.path):
            return self.path[-1]
        # Bỏ qua các điểm đã ở gần
        while self.idx < len(self.path) - 1:
            px, py, _ = self.path[self.idx]
            if math.hypot(px - car.x, py - car.y) < self.lookahead * 0.5:
                self.idx += 1
            else:
                break
        return self.path[self.idx]

    def control(self, car):
        """Trả (throttle, steer_input)."""
        if not self.path:
            return (0, 0)

        # Đến cuối path?
        if self.idx >= len(self.path) - 1:
            gx, gy, _ = self.path[-1]
            if math.hypot(gx - car.x, gy - car.y) < 5.0:
                self.done = True
                return (0, 0)

        tx, ty = self._target(car)[:2]

        # Điểm đích ở local frame của xe
        dx = tx - car.x
        dy = ty - car.y
        c, s = math.cos(-car.theta), math.sin(-car.theta)
        lx = c * dx - s * dy
        ly = s * dx + c * dy

        # Điểm phía trước xe (nếu target phía sau → lùi)
        forward = lx >= 0
        look = max(1e-3, abs(lx))
        alpha = math.atan2(ly, look)

        # Pure pursuit → góc lái mong muốn
        L = car.wheelbase
        ld = max(look, self.lookahead * 0.5)
        steer = math.atan2(2 * L * math.sin(alpha), ld)
        max_s = car.max_steer
        steer = max(-max_s, min(max_s, steer))
        steer_input = steer / max_s

        # Throttle đơn giản
        dist = math.hypot(dx, dy)
        throttle = 1 if forward else -1
        # Giảm tốc khi gần cuối hoặc góc lái lớn
        if abs(steer) > math.radians(25) or dist < 20:
            throttle = 0

        return (throttle, steer_input)