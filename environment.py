from dataclasses import dataclass
from config import WORLD_WIDTH, WORLD_HEIGHT


@dataclass
class CircleObstacle:
    x: float
    y: float
    radius: float


@dataclass
class RectObstacle:
    xmin: float
    ymin: float
    xmax: float
    ymax: float


@dataclass
class DynamicCircleObstacle:
    x: float
    y: float
    radius: float
    vx: float
    vy: float

    def step(self, dt, width, height):
        self.x += self.vx * dt
        self.y += self.vy * dt

        # Nảy lại khi chạm tường
        if self.x - self.radius < 0.0:
            self.x = self.radius
            self.vx = abs(self.vx)
        elif self.x + self.radius > width:
            self.x = width - self.radius
            self.vx = -abs(self.vx)

        if self.y - self.radius < 0.0:
            self.y = self.radius
            self.vy = abs(self.vy)
        elif self.y + self.radius > height:
            self.y = height - self.radius
            self.vy = -abs(self.vy)


class Environment:
    def __init__(self, width=WORLD_WIDTH, height=WORLD_HEIGHT):
        self.width = width
        self.height = height

        # Static circular obstacles
        self.circles = [
            CircleObstacle(5.0, 4.0, 0.8),
            CircleObstacle(8.0, 10.0, 1.0),
            CircleObstacle(14.0, 6.0, 1.2),
            CircleObstacle(15.0, 15.0, 0.7),
            CircleObstacle(6.6, 2.0, 0.5),
            CircleObstacle(5.5, 12.7, 0.65),
        ]

        # Static rectangular obstacles
        self.rectangles = [
            RectObstacle(3.0, 8.0, 5.0, 10.0),
            RectObstacle(9.0, 3.0, 11.0, 6.0),
            RectObstacle(12.0, 11.0, 15.0, 12.5),
            RectObstacle(5.0, 15.0, 8.0, 17.0),
            RectObstacle(2.8, 12.5, 4.2, 13.8),
            RectObstacle(16.0, 3.5, 17.2, 5.0),
        ]

        # Moving obstacles: (x, y, radius, vx, vy)
        self.dynamic_obstacles = [
            DynamicCircleObstacle(11.5, 2.0, 0.4, 0.0, 0.6),
            DynamicCircleObstacle(17.0, 13.5, 0.4, 0.0, 0.25),
            DynamicCircleObstacle(3.0, 7.6, 0.35, 0.6, 0.0),
            DynamicCircleObstacle(6.0, 5.5, 0.35, 0.25, 0.15),
            DynamicCircleObstacle(16.4, 9.5, 0.36, 0.0, 0.25),
            DynamicCircleObstacle(1.8, 11.4, 0.35, 0.25, 0.0),
        ]

    def step(self, dt):
        for obs in self.dynamic_obstacles:
            obs.step(dt, self.width, self.height)

    def get_all_obstacles(self):
        # LiDAR coi vật cản động như hình tròn -> không phải sửa lidar.py
        return self.circles + self.dynamic_obstacles, self.rectangles

    def is_inside(self, x, y):
        return 0.0 <= x <= self.width and 0.0 <= y <= self.height

    def in_collision(self, x, y, radius):
        """Kiểm tra va chạm thật (dùng để đánh giá, robot không biết hàm này)."""
        for c in self.circles + self.dynamic_obstacles:
            if (x - c.x) ** 2 + (y - c.y) ** 2 <= (c.radius + radius) ** 2:
                return True
        for r in self.rectangles:
            dx = max(r.xmin - x, 0.0, x - r.xmax)
            dy = max(r.ymin - y, 0.0, y - r.ymax)
            if dx * dx + dy * dy <= radius * radius:
                return True
        return False