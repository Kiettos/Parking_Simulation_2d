import heapq
import math

from mapark.car import BicycleCar
from mapark.geometry import OBB


# 6 action: (throttle, steer_input) — tiến/lùi × trái/thẳng/phải
MOTIONS = [
    (1, -1), (1, 0), (1, 1),
    (-1, -1), (-1, 0), (-1, 1),
]

STEP_DT = 0.25              # giây mô phỏng mỗi bước
XY_RES = 6.0                # ô lưới cho closed set (pixel)
THETA_RES = math.radians(10)
COLLISION_SAMPLES = 6       # chia mỗi bước thành mấy lần check


class Node:
    __slots__ = ("x", "y", "theta", "v", "steer", "g", "h", "parent", "action")

    def __init__(self, x, y, theta, v, steer, g, h, parent, action):
        self.x = x
        self.y = y
        self.theta = theta
        self.v = v
        self.steer = steer
        self.g = g
        self.h = h
        self.parent = parent
        self.action = action

    def f(self):
        return self.g + self.h


def _key(node):
    return (
        int(node.x // XY_RES),
        int(node.y // XY_RES),
        int((node.theta % (2 * math.pi)) // THETA_RES),
    )


def _heuristic(x, y, theta, goal):
    gx, gy, gtheta = goal
    d = math.hypot(gx - x, gy - y)
    dtheta = abs(((theta - gtheta + math.pi) % (2 * math.pi)) - math.pi)
    return d + 40.0 * dtheta


def _simulate(car, x, y, theta, v, steer, action):
    """Mô phỏng 1 bước, trả list state trung gian để check va chạm."""
    throttle, steer_input = action
    tmp = BicycleCar(x, y, theta,
                     length=car.length, width=car.width,
                     wheelbase=car.wheelbase)
    tmp.v = v
    tmp.steer = steer
    states = []
    sub_dt = STEP_DT / COLLISION_SAMPLES
    for _ in range(COLLISION_SAMPLES):
        st = tmp.propose(sub_dt, throttle, steer_input)
        tmp.commit(st)
        states.append(st)
    return states


def _collides(world, x, y, theta, car):
    obb = OBB(x, y, car.length, car.width, theta)
    return world.car_collides(obb)


def plan(world, goal, max_expand=4000, goal_xy_tol=10.0, goal_theta_tol=math.radians(15)):
    """Hybrid A*. Trả list (x, y, theta) hoặc None."""
    car = world.car
    start = Node(car.x, car.y, car.theta, car.v, car.steer,
                 g=0.0,
                 h=_heuristic(car.x, car.y, car.theta, goal),
                 parent=None, action=None)

    open_heap = [(start.f(), 0, start)]
    closed = set()
    counter = 1
    expanded = 0

    while open_heap and expanded < max_expand:
        _, _, node = heapq.heappop(open_heap)
        k = _key(node)
        if k in closed:
            continue
        closed.add(k)
        expanded += 1

        # Đạt đích? Kiểm tra cả vị trí và góc
        dx = node.x - goal[0]
        dy = node.y - goal[1]
        dtheta = abs(((node.theta - goal[2] + math.pi) % (2 * math.pi)) - math.pi)
        if math.hypot(dx, dy) <= goal_xy_tol and dtheta <= goal_theta_tol:
            return _reconstruct(node)

        for action in MOTIONS:
            states = _simulate(car, node.x, node.y, node.theta,
                               node.v, node.steer, action)
            bad = False
            for (sx, sy, stheta, _sv, _sst) in states:
                if _collides(world, sx, sy, stheta, car):
                    bad = True
                    break
            if bad:
                continue

            sx, sy, stheta, sv, ssteer = states[-1]
            g_new = node.g + STEP_DT
            h_new = _heuristic(sx, sy, stheta, goal)

            child = Node(sx, sy, stheta, sv, ssteer,
                         g=g_new, h=h_new,
                         parent=node, action=action)
            ck = _key(child)
            if ck in closed:
                continue
            heapq.heappush(open_heap, (child.f(), counter, child))
            counter += 1

    return None


def _reconstruct(node):
    path = []
    while node is not None:
        path.append((node.x, node.y, node.theta))
        node = node.parent
    path.reverse()
    return path