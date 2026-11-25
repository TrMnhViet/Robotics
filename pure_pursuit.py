import numpy as np
import math
from config import MAX_SPEED, MAX_YAW_RATE


class PurePursuit:
    def __init__(self, lookahead_distance=1.0, max_speed=MAX_SPEED):
        self.lookahead_distance = lookahead_distance
        self.max_speed = max_speed
        self.max_yaw_rate = MAX_YAW_RATE
        self.current_index = 0

    def reset(self):
        self.current_index = 0

    def find_target(self, position, path):
        if path is None or len(path) == 0:
            return None

        # Chỉ tìm điểm gần nhất từ tiến trình hiện tại trở đi (không quay ngược)
        sub = path[self.current_index:]
        d = np.linalg.norm(sub - position[:2], axis=1)
        self.current_index += int(np.argmin(d))

        # Từ đó tiến lên tìm điểm đủ lookahead
        for i in range(self.current_index, len(path)):
            if np.linalg.norm(path[i] - position[:2]) >= self.lookahead_distance:
                return path[i]
        return path[-1]

    def compute_control(self, state, path):
        target = self.find_target(state[:2], path)

        if target is None:
            return np.array([0.0, 0.0]), None

        dx = target[0] - state[0]
        dy = target[1] - state[1]
        distance = math.hypot(dx, dy)

        if distance < 1e-6:
            return np.array([0.0, 0.0]), target

        target_angle = math.atan2(dy, dx)
        alpha = target_angle - state[2]
        alpha = math.atan2(math.sin(alpha), math.cos(alpha))

        # Lệch hướng quá lớn: quay tại chỗ, không chạy vòng
        if abs(alpha) > math.radians(60):
            v = 0.0
            omega = np.clip(2.0 * alpha, -self.max_yaw_rate, self.max_yaw_rate)
            return np.array([v, omega]), target

        lookahead = max(distance, self.lookahead_distance)
        curvature = 2.0 * math.sin(alpha) / lookahead

        v = self.max_speed * max(0.2, math.cos(alpha))
        omega = np.clip(v * curvature, -self.max_yaw_rate, self.max_yaw_rate)

        return np.array([v, omega]), target