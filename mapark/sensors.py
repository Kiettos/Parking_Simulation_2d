from mapark.geometry import (
    ray_exit_box, ray_obb, rot, vec_add, vec_norm, vec_scale
)

# Cảm biến ego-centric: tên = hướng so với mũi xe
DIRECTIONS_EGO = {
    "F":  (1, 0),
    "FR": (1, 1),
    "R":  (0, 1),
    "BR": (-1, 1),
    "B":  (-1, 0),
    "BL": (-1, -1),
    "L":  (0, -1),
    "FL": (1, -1),
}


def cast_ray(world, origin, direction, max_range):
    d = vec_norm(direction)
    best_t = max_range
    kind = "clear"
    for obs in world.obstacles:
        t = ray_obb(origin, d, obs, best_t)
        if t is not None and t < best_t:
            best_t = t
            kind = "obstacle"
    t_wall = ray_exit_box(origin, d, 0, 0, world.width, world.height, best_t)
    if t_wall is not None and t_wall < best_t:
        best_t = t_wall
        kind = "wall"
    endpoint = vec_add(origin, vec_scale(d, best_t))
    return best_t, endpoint, kind


def sense_all(world, car, max_range=180.0):
    """Trả về dict {dir: (distance, endpoint, kind)} theo hướng thân xe."""
    origin = (car.x, car.y)
    out = {}
    for name, local_vec in DIRECTIONS_EGO.items():
        world_vec = rot(local_vec, car.theta)
        out[name] = cast_ray(world, origin, world_vec, max_range)
    return out