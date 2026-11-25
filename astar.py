import heapq
import math
import numpy as np

from config import ROBOT_RADIUS, ASTAR_SAFETY_MARGIN
from occupancy_grid import UNKNOWN, FREE, OCCUPIED


class AStarPlanner:
    def __init__(self, occupancy_grid):
        self.map = occupancy_grid
        self.resolution = occupancy_grid.resolution
        self.width = occupancy_grid.cols
        self.height = occupancy_grid.rows

        # Inflate obstacles to account for robot size
        self.grid = occupancy_grid.inflate_obstacles(
            ROBOT_RADIUS + ASTAR_SAFETY_MARGIN
        )

        # 8 possible movement directions
        self.motion = [
            (-1, 0, 1.0),
            (1, 0, 1.0),
            (0, -1, 1.0),
            (0, 1, 1.0),
            (-1, -1, math.sqrt(2)),
            (-1, 1, math.sqrt(2)),
            (1, -1, math.sqrt(2)),
            (1, 1, math.sqrt(2))
        ]

        # Penalty for traveling through unknown cells
        self.unknown_cost = 1.5

    def world_to_grid(self, x, y):
        col = int(math.floor(x / self.resolution))
        row = int(math.floor(y / self.resolution))
        return col, row

    def grid_to_world(self, col, row):
        x = (col + 0.5) * self.resolution
        y = (row + 0.5) * self.resolution
        return x, y

    def is_valid(self, col, row):
        if not (0 <= col < self.width and 0 <= row < self.height):
            return False

        return self.grid[row, col] != OCCUPIED

    def clear_around(self, node, radius_cells):
        """
        Mo khoang trong quanh node (thuong la vi tri robot).
        Chi xoa cac o bi chan do inflate, giu nguyen vat can that
        (o OCCUPIED trong ban do goc self.map.grid).
        """
        col0, row0 = node
        for dr in range(-radius_cells, radius_cells + 1):
            for dc in range(-radius_cells, radius_cells + 1):
                if dr * dr + dc * dc > radius_cells * radius_cells:
                    continue
                r, c = row0 + dr, col0 + dc
                if 0 <= c < self.width and 0 <= r < self.height:
                    if self.map.grid[r, c] != OCCUPIED:
                        self.grid[r, c] = self.map.grid[r, c]

    def heuristic(self, a, b):
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        return max(dx, dy) + (math.sqrt(2) - 1) * min(dx, dy)

    def reconstruct_path(self, came_from, current):
        path = [current]

        while current in came_from:
            current = came_from[current]
            path.append(current)

        path.reverse()

        return np.array([
            self.grid_to_world(col, row)
            for col, row in path
        ])

    def plan(self, start, goal):
        start_node = self.world_to_grid(start[0], start[1])
        goal_node = self.world_to_grid(goal[0], goal[1])

        # Robot co the da nam trong vung inflate: mo khoang trong quanh start
        if not self.is_valid(*start_node):
            self.clear_around(start_node, radius_cells=4)

        if not self.is_valid(*start_node):
            print("A*: Start cell is occupied or out of bounds.")
            return None

        if not self.is_valid(*goal_node):
            print("A*: Goal cell is occupied or out of bounds.")
            return None

        open_list = []
        heapq.heappush(
            open_list,
            (self.heuristic(start_node, goal_node), 0.0, start_node)
        )

        came_from = {}
        g_score = {start_node: 0.0}
        closed_set = set()

        while open_list:
            _, current_cost, current = heapq.heappop(open_list)

            if current in closed_set:
                continue

            if current == goal_node:
                grid_path = self.reconstruct_path(came_from, current)

                path = np.vstack((
                    np.asarray(start, dtype=float),
                    grid_path,
                    np.asarray(goal, dtype=float)
                ))

                return path

            closed_set.add(current)

            for dx, dy, move_cost in self.motion:
                neighbor = (
                    current[0] + dx,
                    current[1] + dy
                )

                if not self.is_valid(*neighbor):
                    continue

                # Prevent diagonal movement through obstacle corners
                if dx != 0 and dy != 0:
                    if not self.is_valid(current[0] + dx, current[1]):
                        continue
                    if not self.is_valid(current[0], current[1] + dy):
                        continue

                if neighbor in closed_set:
                    continue

                # Prefer known free cells over unknown cells
                cell_state = self.grid[neighbor[1], neighbor[0]]
                cell_penalty = (
                    self.unknown_cost if cell_state == UNKNOWN else 1.0
                )

                tentative_g = (
                    g_score[current]
                    + move_cost * self.resolution * cell_penalty
                )

                if tentative_g < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g

                    f_score = (
                        tentative_g
                        + self.heuristic(neighbor, goal_node) * self.resolution
                    )

                    heapq.heappush(
                        open_list,
                        (f_score, tentative_g, neighbor)
                    )

        return None