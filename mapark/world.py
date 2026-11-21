from mapark.sensors import sense_all
from Unimportant.geometry import obb_intersect


class World:
    def __init__(self, width, height):
        self.width = float(width)
        self.height = float(height)
        self.car = None
        self.static_obstacles = []     # vật cản cố định (tường, chướng ngại)
        self.decoys = []               # xe decoy (đổi mỗi N episode)
        self.decoy_slots = []          # index các slot đang bị decoy chiếm
        self.obstacles = []            # static + decoys, dùng cho raycast/va chạm
        self.walls = []                # tường từ layout JSON (để vẽ riêng)
        self.slots = []
        self.sensors = {}
        self.planned_path = []      # list (x, y, theta) từ Hybrid A*
        self.planning = False

        # Metadata map JSON (None nếu dựng world kiểu cũ)
        self.cell_size = None
        self.cols = None
        self.rows = None
        self.start_pose = None

        # Slot đang được chọn làm đích (cho render)
        self.target_index = 0

    def add_obstacle(self, obb):
        self.static_obstacles.append(obb)
        self.obstacles.append(obb)

    def add_wall(self, obb):
        self.walls.append(obb)
        self.add_obstacle(obb)

    def add_slot(self, obb):
        self.slots.append(obb)

    def set_decoys(self, decoys):
        """Thay toàn bộ xe decoy; rebuild danh sách vật cản."""
        self.decoys = list(decoys)
        self.decoy_slots = []
        self.obstacles = self.static_obstacles + self.decoys

    def free_slot_indices(self):
        """Các slot không bị xe decoy chiếm."""
        free = []
        for i, slot in enumerate(self.slots):
            if not any(obb_intersect(slot, d) for d in self.decoys):
                free.append(i)
        return free

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
