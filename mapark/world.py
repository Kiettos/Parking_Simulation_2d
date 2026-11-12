from mapark.sensors import sense_all
from mapark.geometry import obb_intersect


class World:
    def __init__(self, width, height):
        self.width = float(width)
        self.height = float(height)
        self.car = None
        self.obstacles = []
        self.slots = []
        self.sensors = {}
        self.planned_path = []      # list (x, y, theta) từ Hybrid A*
        self.planning = False

    def add_obstacle(self, obb):
        self.obstacles.append(obb)

    def add_slot(self, obb):
        self.slots.append(obb)

    def car_collides(self, obb):
        for c in obb.corners():
            if c[0] < 0 or c[0] > self.width or c[1] < 0 or c[1] > self.height:
                return True
        for obs in self.obstacles:
            if obb_intersect(obb, obs):
                return True
        return False

    def step(self, dt, throttle, steer_input):
        car = self.car
        old = (car.x, car.y, car.theta, car.v, car.steer)
        car.commit(car.propose(dt, throttle, steer_input))
        if self.car_collides(car.obb()):
            car.commit(old)
            car.v = 0.0
        self.sensors = sense_all(self, car)