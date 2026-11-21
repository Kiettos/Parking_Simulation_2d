import math
import random

import pygame

from Unimportant.geometry import OBB
from Unimportant.hybrid_astar import plan
from mapark.path_follower import PathFollower

WHITE = (245, 245, 245)
GRID = (220, 220, 220)
WALL = (70, 70, 80)
OBST = (90, 90, 110)
SLOT = (90, 180, 220)
TARGET = (255, 150, 40)
TARGET_FILL = (255, 228, 195)
CAR = (60, 130, 220)
CAR_FRONT = (255, 220, 90)
DECOY = (150, 150, 160)
DECOY_MARK = (110, 110, 120)
PATH_COLOR = (255, 140, 0)
SENSOR_COLORS = {
    "clear":    (80, 200, 120),
    "obstacle": (220, 80, 80),
    "wall":     (200, 120, 220),
}


def draw_obb(surface, obb, color, fill=None, width=2):
    pts = obb.corners()
    if fill is not None:
        pygame.draw.polygon(surface, fill, pts)
    pygame.draw.polygon(surface, color, pts, width)


def _free_slots(world):
    occupied = set(getattr(world, "decoy_slots", []))
    return [i for i in range(len(world.slots)) if i not in occupied]


def _pick_target(world):
    free = _free_slots(world)
    return random.choice(free) if free else 0


def _spawn_decoys(world, count=5):
    """Đặt `count` xe decoy vào slot ngẫu nhiên (để demo)."""
    n = len(world.slots)
    if n <= 1:
        world.set_decoys([])
        return
    k = min(count, n - 1)
    ids = sorted(random.sample(range(n), k))
    car = world.car
    decoys = [
        OBB(world.slots[i].cx, world.slots[i].cy, car.length, car.width, world.slots[i].theta)
        for i in ids
    ]
    world.set_decoys(decoys)
    world.decoy_slots = ids


def _hex(c, default=(200, 200, 200)):
    """'#rrggbb' -> (r, g, b)."""
    if not c:
        return default
    c = c.lstrip("#")
    return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))


def _draw_tiles(surface, world):
    """Tô màu từng ô tile + bush/lamp theo legend trong config."""
    leg = world.legend
    ts = world.cell_size
    for r, row in enumerate(world.tiles_raw):
        for c, ch in enumerate(row):
            spec = leg["tiles"].get(ch)
            if not spec:
                continue
            rect = pygame.Rect(int(c * ts), int(r * ts), int(ts), int(ts))
            pygame.draw.rect(surface, _hex(spec.get("color")), rect)
            if spec.get("strokeColor"):
                pygame.draw.rect(surface, _hex(spec["strokeColor"]), rect, 1)

    o = world.obj_size
    for r, row in enumerate(world.objects_raw):
        for c, ch in enumerate(row):
            spec = leg["objects"].get(ch)
            if spec:
                x = c * ts + (ts - o) / 2
                y = r * ts + (ts - o) / 2
                pygame.draw.rect(surface, _hex(spec.get("color")), (x, y, o, o))


def draw_world(surface, world):
    surface.fill(WHITE)

    tiled = bool(
        getattr(world, "tiles_raw", None) and getattr(world, "legend", None)
    )

    if tiled:
        _draw_tiles(surface, world)
    else:
        step = int(getattr(world, "cell_size", None) or 40)
        for x in range(0, int(world.width) + 1, step):
            pygame.draw.line(surface, GRID, (x, 0), (x, world.height))
        for y in range(0, int(world.height) + 1, step):
            pygame.draw.line(surface, GRID, (0, y), (world.width, y))

    # Slots
    target_idx = getattr(world, "target_index", 0)
    for i, slot in enumerate(world.slots):
        if i == target_idx:
            draw_obb(surface, slot, TARGET, fill=TARGET_FILL, width=3)
        else:
            draw_obb(surface, slot, SLOT, fill=None, width=2)

    walls = getattr(world, "walls", [])
    wall_ids = {id(w) for w in walls}
    decoy_ids = {id(d) for d in getattr(world, "decoys", [])}

    # Có tile thì tường (G/H) và bush/lamp đã được tô màu ở _draw_tiles,
    # nên không vẽ lại kẻo đè mất màu.
    if not tiled:
        for w in walls:
            draw_obb(surface, w, WALL, fill=WALL)
        for obs in world.obstacles:
            if id(obs) in wall_ids or id(obs) in decoy_ids:
                continue
            draw_obb(surface, obs, OBST, fill=OBST)

    # Decoys
    for decoy in getattr(world, "decoys", []):
        draw_obb(surface, decoy, DECOY, fill=DECOY)
        c = decoy.corners()
        pygame.draw.line(surface, DECOY_MARK, c[0], c[2], 1)
        pygame.draw.line(surface, DECOY_MARK, c[1], c[3], 1)

    # Path đã plan
    if world.planned_path:
        pts = [(int(x), int(y)) for x, y, _ in world.planned_path]
        if len(pts) >= 2:
            pygame.draw.lines(surface, PATH_COLOR, False, pts, 3)
        for i in range(0, len(pts), max(1, len(pts) // 20)):
            pygame.draw.circle(surface, PATH_COLOR, pts[i], 3)

    # Sensors
    car = world.car
    origin = (car.x, car.y)
    for _name, (dist, endpoint, kind) in world.sensors.items():
        color = SENSOR_COLORS[kind]
        pygame.draw.line(surface, color, origin, endpoint, 2)
        pygame.draw.circle(surface, color, (int(endpoint[0]), int(endpoint[1])), 3)

    # Xe (viền trắng để không chìm vào nền tối)
    car_col = _hex(getattr(world, "car_color", None), CAR)
    draw_obb(surface, car.obb(), (255, 255, 255), fill=car_col, width=1)
    fx = car.x + math.cos(car.theta) * car.length * 0.35
    fy = car.y + math.sin(car.theta) * car.length * 0.35
    pygame.draw.circle(surface, CAR_FRONT, (int(fx), int(fy)), 4)


def run(world):
    pygame.init()
    screen = pygame.display.set_mode((int(world.width), int(world.height)))
    pygame.display.set_caption(
        "ParkSim2D — Hybrid A*  |  SPACE = plan  ENTER = follow  D = decoys  R = reset"
    )
    font = pygame.font.SysFont("consolas", 14)
    clock = pygame.time.Clock()
    dt = 1.0 / 60.0
    start = getattr(world, "start_pose", None) or (
        world.car.x,
        world.car.y,
        world.car.theta,
    )
    car0 = (float(start[0]), float(start[1]), float(start[2]))

    follower = PathFollower(lookahead=90)
    auto_follow = False
    status = "Drive manually (arrow keys), SPACE to plan"

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    world.car.commit((car0[0], car0[1], car0[2], 0.0, 0.0))
                    world.planned_path = []
                    world.set_decoys([])
                    world.target_index = 0
                    follower.set_path([])
                    auto_follow = False
                    status = "Reset"
                elif event.key == pygame.K_d:
                    _spawn_decoys(world, count=5)
                    world.planned_path = []
                    follower.set_path([])
                    auto_follow = False
                    world.target_index = _pick_target(world)
                    status = (
                        f"Decoys at slots {sorted(world.decoy_slots)};"
                        f" target slot {world.target_index}"
                    )
                elif event.key == pygame.K_SPACE:
                    if not world.slots:
                        status = "No slots configured"
                    else:
                        idx = world.target_index
                        if idx not in _free_slots(world):
                            idx = _pick_target(world)
                            world.target_index = idx
                        slot = world.slots[idx]
                        goal = (slot.cx, slot.cy, slot.theta)
                        status = "Planning..."
                        pygame.display.set_caption(status)
                        path = plan(world, goal, max_expand=6000)
                        if path is None:
                            status = "Plan FAILED"
                        else:
                            world.planned_path = path
                            follower.set_path(path)
                            status = f"Plan OK: {len(path)} nodes -> slot {idx}"
                            auto_follow = True
                elif event.key == pygame.K_RETURN:
                    if world.planned_path:
                        follower.set_path(world.planned_path)
                        auto_follow = True
                        status = "Following path"

        if auto_follow:
            throttle, steer_input = follower.control(world.car)
            if follower.done:
                throttle, steer_input = 0, 0
                auto_follow = False
                status = "Parked"
        else:
            keys = pygame.key.get_pressed()
            throttle = 0
            if keys[pygame.K_UP]:
                throttle = 1
            elif keys[pygame.K_DOWN]:
                throttle = -1
            steer_input = 0
            if keys[pygame.K_RIGHT]:
                steer_input = 1
            elif keys[pygame.K_LEFT]:
                steer_input = -1

        world.step(dt, throttle, steer_input)
        draw_world(screen, world)

        hud = font.render(status, True, (255, 255, 255), (0, 0, 0))
        screen.blit(hud, (10, 10))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()