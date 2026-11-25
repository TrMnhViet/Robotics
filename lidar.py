
import numpy as np
import math
from config import LIDAR_NUM_RAYS, LIDAR_MAX_RANGE
from environment import Environment


class LiDAR:
    def __init__(self, num_rays=LIDAR_NUM_RAYS, max_range=LIDAR_MAX_RANGE):
        self.num_rays = num_rays
        self.max_range = max_range
        self.angles = np.linspace(-math.pi, math.pi, num_rays, endpoint=False)

    def ray_circle_intersection(self, origin, direction, circle):
        center = np.array([circle.x, circle.y], dtype=float)
        oc = origin - center

        a = np.dot(direction, direction)
        b = 2.0 * np.dot(direction, oc)
        c = np.dot(oc, oc) - circle.radius ** 2

        discriminant = b * b - 4.0 * a * c
        if discriminant < 0:
            return None

        sqrt_d = math.sqrt(discriminant)
        t1 = (-b - sqrt_d) / (2.0 * a)
        t2 = (-b + sqrt_d) / (2.0 * a)

        valid = [t for t in (t1, t2) if t > 1e-8]
        return min(valid) if valid else None

    def ray_rectangle_intersection(self, origin, direction, rect):
        t_min = -float("inf")
        t_max = float("inf")

        bounds = [
            (rect.xmin, rect.xmax),
            (rect.ymin, rect.ymax)
        ]

        for i, (lower, upper) in enumerate(bounds):
            if abs(direction[i]) < 1e-10:
                if origin[i] < lower or origin[i] > upper:
                    return None
            else:
                t1 = (lower - origin[i]) / direction[i]
                t2 = (upper - origin[i]) / direction[i]

                t_min = max(t_min, min(t1, t2))
                t_max = min(t_max, max(t1, t2))

                if t_min > t_max:
                    return None

        if t_max <= 1e-8:
            return None

        return t_min if t_min > 1e-8 else t_max

    def ray_wall_intersection(self, origin, direction, env):
        intersections = []

        # Left and right walls
        if abs(direction[0]) > 1e-10:
            for x_wall in (0.0, env.width):
                t = (x_wall - origin[0]) / direction[0]
                if t > 1e-8:
                    y = origin[1] + t * direction[1]
                    if -1e-8 <= y <= env.height + 1e-8:
                        intersections.append(t)

        # Bottom and top walls
        if abs(direction[1]) > 1e-10:
            for y_wall in (0.0, env.height):
                t = (y_wall - origin[1]) / direction[1]
                if t > 1e-8:
                    x = origin[0] + t * direction[0]
                    if -1e-8 <= x <= env.width + 1e-8:
                        intersections.append(t)

        return min(intersections) if intersections else None

    def scan(self, robot_state, env):
        """Quét 360 tia dạng mảng: tính song song mọi hướng LiDAR.

        Vẫn sử dụng đúng các giao hình học với circle/rectangle/wall và
        chỉ trả về tia chạm vật cản ở khoảng cách < max_range như trước.
        """
        x, y, yaw = robot_state[:3]
        world_angles = yaw + self.angles
        dx = np.cos(world_angles)
        dy = np.sin(world_angles)
        nearest = np.full(self.num_rays, self.max_range, dtype=float)
        eps = 1e-8

        circles, rectangles = env.get_all_obstacles()

        # Mỗi vật cản tròn: kiểm tra 360 tia đồng thời.
        a = dx * dx + dy * dy
        for circle in circles:
            ox = x - circle.x
            oy = y - circle.y
            b = 2.0 * (dx * ox + dy * oy)
            c = ox * ox + oy * oy - circle.radius * circle.radius
            disc = b * b - 4.0 * a * c
            root = np.sqrt(np.maximum(disc, 0.0))
            t1 = (-b - root) / (2.0 * a)
            t2 = (-b + root) / (2.0 * a)
            t = np.where(t1 > eps, t1, np.where(t2 > eps, t2, np.inf))
            nearest = np.minimum(nearest, np.where(disc >= 0.0, t, np.inf))

        # Hình chữ nhật: slab intersection được vector hóa theo hướng tia.
        def slab(origin, direction, lower, upper):
            parallel = np.abs(direction) < 1e-10
            t_low = np.empty_like(direction)
            t_high = np.empty_like(direction)
            np.divide(lower - origin, direction, out=t_low, where=~parallel)
            np.divide(upper - origin, direction, out=t_high, where=~parallel)
            t_enter = np.where(parallel, -np.inf, np.minimum(t_low, t_high))
            t_exit = np.where(parallel, np.inf, np.maximum(t_low, t_high))
            blocked = parallel & ((origin < lower) | (origin > upper))
            return t_enter, t_exit, blocked

        for rect in rectangles:
            ex, lx, bx = slab(x, dx, rect.xmin, rect.xmax)
            ey, ly, by = slab(y, dy, rect.ymin, rect.ymax)
            enter = np.maximum(ex, ey)
            leave = np.minimum(lx, ly)
            t = np.where(enter > eps, enter, leave)
            valid = (~bx) & (~by) & (enter <= leave) & (leave > eps)
            nearest = np.minimum(nearest, np.where(valid, t, np.inf))

        # Tường bao môi trường: 4 đoạn thẳng, không phải tia vô hạn.
        for wall_x in (0.0, env.width):
            active = np.abs(dx) > 1e-10
            t = np.full_like(dx, np.inf)
            np.divide(wall_x - x, dx, out=t, where=active)
            y_hit = y + t * dy
            valid = active & (t > eps) & (y_hit >= -eps) & (y_hit <= env.height + eps)
            nearest = np.minimum(nearest, np.where(valid, t, np.inf))

        for wall_y in (0.0, env.height):
            active = np.abs(dy) > 1e-10
            t = np.full_like(dy, np.inf)
            np.divide(wall_y - y, dy, out=t, where=active)
            x_hit = x + t * dx
            valid = active & (t > eps) & (x_hit >= -eps) & (x_hit <= env.width + eps)
            nearest = np.minimum(nearest, np.where(valid, t, np.inf))

        # Bản cũ chỉ coi tia chạm vật cản khi distance < max_range.
        valid = nearest < self.max_range
        ranges = np.where(valid, nearest, np.inf)
        points = np.column_stack((x + ranges[valid] * dx[valid],
                                  y + ranges[valid] * dy[valid]))
        return {
            "angles": self.angles.copy(),
            "world_angles": world_angles,
            "ranges": ranges,
            "points": points,
            "valid": valid,
            "max_range": self.max_range,
        }
