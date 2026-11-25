
import numpy as np


class PathTracker:
    def __init__(self, path, lookahead_distance=1.0):
        self.path = np.asarray(path, dtype=float)
        self.lookahead_distance = lookahead_distance
        self.current_index = 0

    def get_target(self, robot_position):
        # Find nearest waypoint, starting from current progress
        remaining = self.path[self.current_index:]
        distances = np.linalg.norm(
            remaining - robot_position[:2],
            axis=1
        )

        nearest_offset = int(np.argmin(distances))
        self.current_index += nearest_offset

        # Move forward along path by lookahead distance
        target_index = self.current_index
        accumulated_distance = 0.0

        while target_index < len(self.path) - 1:
            segment = np.linalg.norm(
                self.path[target_index + 1] - self.path[target_index]
            )

            accumulated_distance += segment
            target_index += 1

            if accumulated_distance >= self.lookahead_distance:
                break

        return self.path[target_index].copy()

    def is_finished(self):
        return self.current_index >= len(self.path) - 1
