import math

import pygame

from mapark.hybrid_astar import plan
from mapark.path_follower import PathFollower
from mapark.world import World

WHITE = (245, 245, 245)
GRID = (220, 220, 220)
OBST = (90, 90, 110)
SLOT = (90, 180, 220)
CAR = (60, 130, 220)
CAR_FRONT = (255, 220, 90)
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


def draw_world(surface, world):
    surface.fill(WHITE)
    step = 40
    for x in range(0, int(world.width) + 1, step):
        pygame.draw.line(surface, GRID, (x, 0), (x, world.height))
    for y in range(0, int(world.height) + 1, step):
        pygame.draw.line(surface, GRID, (0, y), (world.width, y))

    for slot in world.slots:
        draw_obb(surface, slot, SLOT, fill=None, width=2)
    for obs in world.obstacles:
        draw_obb(surface, obs, OBST, fill=OBST)

    # Path đã plan
    if world.planned_path:
        pts = [(int(x), int(y)) for x, y, _ in world.planned_path]
        if len(pts) >= 2:
            pygame.draw.lines(surface, PATH_COLOR, False, pts, 3)
        for i in range(0, len(pts), max(1, len(pts) // 20)):
            pygame.draw.circle(surface, PATH_COLOR, pts[i], 3)

    car = world.car
    origin = (car.x, car.y)
    for _name, (dist, endpoint, kind) in world.sensors.items():
        color = SENSOR_COLORS[kind]
        pygame.draw.line(surface, color, origin, endpoint, 2)
        pygame.draw.circle(surface, color, (int(endpoint[0]), int(endpoint[1])), 3)

    draw_obb(surface, car.obb(), CAR, fill=CAR)
    fx = car.x + math.cos(car.theta) * car.length * 0.35
    fy = car.y + math.sin(car.theta) * car.length * 0.35
    pygame.draw.circle(surface, CAR_FRONT, (int(fx), int(fy)), 4)


def run(world):
    pygame.init()
    screen = pygame.display.set_mode((int(world.width), int(world.height)))
    pygame.display.set_caption(
        "ParkSim2D — Hybrid A*  |  SPACE = plan  ENTER = follow  R = reset"
    )
    font = pygame.font.SysFont("consolas", 14)
    clock = pygame.time.Clock()
    dt = 1.0 / 60.0
    car0 = (world.car.x, world.car.y, world.car.theta)

    follower = PathFollower(lookahead=45.0)
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
                    follower.set_path([])
                    auto_follow = False
                    status = "Reset"
                elif event.key == pygame.K_SPACE:
                    if not world.slots:
                        status = "No slots configured"
                    else:
                        slot = world.slots[0]
                        goal = (slot.cx, slot.cy, slot.theta)
                        status = "Planning..."
                        pygame.display.set_caption(status)
                        path = plan(world, goal, max_expand=6000)
                        if path is None:
                            status = "Plan FAILED"
                        else:
                            world.planned_path = path
                            follower.set_path(path)
                            status = f"Plan OK: {len(path)} nodes"
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

        hud = font.render(status, True, (30, 30, 30))
        screen.blit(hud, (10, 10))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()