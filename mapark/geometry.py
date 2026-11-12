import math


def vec_add(a, b):
    return (a[0] + b[0], a[1] + b[1])


def vec_sub(a, b):
    return (a[0] - b[0], a[1] - b[1])


def vec_scale(a, s):
    return (a[0] * s, a[1] * s)


def vec_dot(a, b):
    return a[0] * b[0] + a[1] * b[1]


def vec_len(a):
    return math.hypot(a[0], a[1])


def vec_norm(a):
    l = vec_len(a)
    return (a[0] / l, a[1] / l) if l > 1e-9 else (0.0, 0.0)


def rot(p, theta):
    c, s = math.cos(theta), math.sin(theta)
    return (c * p[0] - s * p[1], s * p[0] + c * p[1])


class OBB:
    """Hộp chữ nhật có góc xoay (Oriented Bounding Box)."""

    def __init__(self, cx, cy, w, h, theta=0.0):
        self.cx = float(cx)
        self.cy = float(cy)
        self.w = float(w)
        self.h = float(h)
        self.theta = float(theta)

    def corners(self):
        hw, hh = self.w / 2, self.h / 2
        local = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
        c = (self.cx, self.cy)
        return [vec_add(c, rot(p, self.theta)) for p in local]

    def axes(self):
        c, s = math.cos(self.theta), math.sin(self.theta)
        return [(c, s), (-s, c)]


def _project(points, axis):
    vals = [vec_dot(p, axis) for p in points]
    return (min(vals), max(vals))


def obb_intersect(a, b):
    """SAT test — hai OBB chồng nhau?"""
    for axis in a.axes() + b.axes():
        amin, amax = _project(a.corners(), axis)
        bmin, bmax = _project(b.corners(), axis)
        if amax < bmin or bmax < amin:
            return False
    return True


def ray_obb(origin, direction, obb, max_t):
    """Tia từ origin theo direction cắt obb. Trả t nhỏ nhất hoặc None."""
    o = vec_sub(origin, (obb.cx, obb.cy))
    o = rot(o, -obb.theta)
    d = rot(direction, -obb.theta)
    hw, hh = obb.w / 2, obb.h / 2
    tmin, tmax = 0.0, max_t
    for ov, dv, hv in ((o[0], d[0], hw), (o[1], d[1], hh)):
        if abs(dv) < 1e-9:
            if abs(ov) > hv:
                return None
        else:
            t1 = (-hv - ov) / dv
            t2 = (hv - ov) / dv
            if t1 > t2:
                t1, t2 = t2, t1
            tmin = max(tmin, t1)
            tmax = min(tmax, t2)
            if tmin > tmax:
                return None
    return tmin if tmin <= max_t else None


def ray_exit_box(origin, direction, x0, y0, x1, y1, max_t):
    """Tia xuất phát từ trong hộp, tìm điểm thoát khỏi biên."""
    t = max_t
    if direction[0] > 1e-9:
        t = min(t, (x1 - origin[0]) / direction[0])
    elif direction[0] < -1e-9:
        t = min(t, (x0 - origin[0]) / direction[0])
    if direction[1] > 1e-9:
        t = min(t, (y1 - origin[1]) / direction[1])
    elif direction[1] < -1e-9:
        t = min(t, (y0 - origin[1]) / direction[1])
    return t if t <= max_t else None