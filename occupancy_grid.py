import numpy as np
from scipy.ndimage import binary_dilation
from config import WORLD_WIDTH, WORLD_HEIGHT

UNKNOWN = -1
FREE = 0
OCCUPIED = 1


class OccupancyGrid:
    """
    Occupancy grid dạng log-odds:
      - ô trúng tia LiDAR  -> cộng l_occ
      - ô tia đi xuyên qua -> trừ l_free
    Nhờ vậy khi vật cản động đi khỏi, các ô cũ sẽ được xóa sau vài lần quét
    (hết "vệt ma"), và nhiễu 1 lần quét không làm đổi bản đồ ngay.
    """

    def __init__(self, width=WORLD_WIDTH, height=WORLD_HEIGHT, resolution=0.2,
                 l_occ=1.0, l_free=0.4, l_min=-2.0, l_max=2.0,
                 occ_threshold=0.5):
        self.width = width
        self.height = height
        self.resolution = resolution
        self.cols = int(np.ceil(width / resolution))
        self.rows = int(np.ceil(height / resolution))

        self.l_occ = l_occ
        self.l_free = l_free
        self.l_min = l_min
        self.l_max = l_max
        self.occ_threshold = occ_threshold

        self.log_odds = np.zeros((self.rows, self.cols), dtype=np.float32)
        self.visited = np.zeros((self.rows, self.cols), dtype=bool)
        self.grid = np.full((self.rows, self.cols), UNKNOWN, dtype=np.int8)

    def world_to_grid(self, x, y):
        col = int(x / self.resolution)
        row = int(y / self.resolution)
        return row, col

    def grid_to_world(self, row, col):
        x = (col + 0.5) * self.resolution
        y = (row + 0.5) * self.resolution
        return x, y

    def in_bounds(self, row, col):
        return 0 <= row < self.rows and 0 <= col < self.cols

    def bresenham(self, start, end):
        x0, y0 = start
        x1, y1 = end
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            points.append((y0, x0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

        return points

    def update(self, robot_state, scan):
        robot_x, robot_y = robot_state[:2]
        start_row, start_col = self.world_to_grid(robot_x, robot_y)

        angles = scan["world_angles"]
        ranges = scan["ranges"]
        valid = scan["valid"]
        max_range = scan.get("max_range", None)
        has_obstacle = np.asarray(valid & np.isfinite(ranges), dtype=bool)

        if max_range is None:
            use_ray = has_obstacle
            lengths = np.where(has_obstacle, ranges, 0.0)
        else:
            use_ray = np.ones(len(angles), dtype=bool)
            lengths = np.where(has_obstacle, ranges, max_range)

        # Tính đầu mút mọi tia một lần, tránh 720 lần gọi hàm sin/cos Python.
        endpoint_x = robot_x + lengths * np.cos(angles)
        endpoint_y = robot_y + lengths * np.sin(angles)
        end_cols = np.trunc(endpoint_x / self.resolution).astype(np.int64)
        end_rows = np.trunc(endpoint_y / self.resolution).astype(np.int64)

        # Các update cùng ô phải giữ đúng thứ tự từng tia, để log-odds
        # và kết quả threshold không bị thay đổi bởi nhóm phép cộng.
        indices = []
        values = []
        cols = self.cols
        rows = self.rows
        for i in range(len(angles)):
            if not use_ray[i]:
                continue
            end_row, end_col = int(end_rows[i]), int(end_cols[i])
            cells = self.bresenham((start_col, start_row), (end_col, end_row))
            free_cells = cells[:-1] if has_obstacle[i] else cells
            for row, col in free_cells:
                if 0 <= row < rows and 0 <= col < cols:
                    indices.append(row * cols + col)
                    values.append(-self.l_free)
            if has_obstacle[i] and 0 <= end_row < rows and 0 <= end_col < cols:
                indices.append(end_row * cols + end_col)
                values.append(self.l_occ)

        if indices:
            index = np.asarray(indices, dtype=np.intp)
            flat = self.log_odds.ravel()
            np.add.at(flat, index, np.asarray(values, dtype=np.float32))
            self.visited.ravel()[index] = True

        np.clip(self.log_odds, self.l_min, self.l_max, out=self.log_odds)
        self._refresh_grid()

    def _refresh_grid(self):
        # Tái sử dụng buffer, không tạo mảng mới 100x100 ở mọi lần quét.
        self.grid.fill(UNKNOWN)
        self.grid[self.visited] = FREE
        self.grid[self.visited & (self.log_odds > self.occ_threshold)] = OCCUPIED

    def inflate_obstacles(self, robot_radius):
        # Dùng binary_dilation thay vòng lặp Python -> nhanh hơn hàng chục lần
        r = int(np.ceil(robot_radius / self.resolution))
        yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
        disk = (xx * xx + yy * yy) <= r * r

        occupied = self.grid == OCCUPIED
        inflated = self.grid.copy()
        inflated[binary_dilation(occupied, structure=disk)] = OCCUPIED
        return inflated

    def get_map(self):
        return self.grid.copy()