import numpy as np
import math
from config import (
    ROBOT_RADIUS, MAX_SPEED, MIN_SPEED, MAX_YAW_RATE, MAX_ACCEL,
    MAX_DELTA_YAW_RATE, V_RESOLUTION, YAW_RATE_RESOLUTION, DT,
    PREDICT_TIME, TO_GOAL_COST_GAIN, SPEED_COST_GAIN, OBSTACLE_COST_GAIN
)


class DWA:
    def __init__(self):
        self.robot_radius = ROBOT_RADIUS
        self.max_speed = MAX_SPEED
        self.min_speed = MIN_SPEED
        self.max_yaw_rate = MAX_YAW_RATE
        self.max_accel = MAX_ACCEL
        self.max_delta_yaw_rate = MAX_DELTA_YAW_RATE
        self.v_resolution = V_RESOLUTION
        self.yaw_rate_resolution = YAW_RATE_RESOLUTION
        self.dt = DT
        self.predict_time = PREDICT_TIME
        self.to_goal_gain = TO_GOAL_COST_GAIN
        self.speed_gain = SPEED_COST_GAIN
        self.obstacle_gain = OBSTACLE_COST_GAIN
        self.path_gain = 2.0

        # Khoảng cách tối thiểu (tính từ mép robot) coi như va chạm
        self.collision_margin = 0.05
        # Small buffer for preselecting obstacles before swept checks.
        # The measured LiDAR points are on obstacle surfaces; collision
        # checks inflate these points by the robot's radius exactly once.
        self.selection_margin = 0.05

    # ------------------------------------------------------------------
    def motion(self, state, control):
        x = state.copy()
        v, omega = control
        x[2] += omega * self.dt
        x[2] = (x[2] + np.pi) % (2 * np.pi) - np.pi
        x[0] += v * np.cos(x[2]) * self.dt
        x[1] += v * np.sin(x[2]) * self.dt
        x[3] = v
        x[4] = omega
        return x

    def dynamic_window(self, state):
        vs = [self.min_speed, self.max_speed, -self.max_yaw_rate, self.max_yaw_rate]
        vd = [
            state[3] - self.max_accel * self.dt,
            state[3] + self.max_accel * self.dt,
            state[4] - self.max_delta_yaw_rate * self.dt,
            state[4] + self.max_delta_yaw_rate * self.dt
        ]
        return [
            max(vs[0], vd[0]), min(vs[1], vd[1]),
            max(vs[2], vd[2]), min(vs[3], vd[3])
        ]

    def predict_trajectory(self, state, control):
        trajectory = [state.copy()]
        current_state = state.copy()
        steps = int(self.predict_time / self.dt)
        for _ in range(steps):
            current_state = self.motion(current_state, control)
            trajectory.append(current_state.copy())
        return np.array(trajectory)

    @staticmethod
    def _approach_zero(value, max_change):
        """Limit deceleration without reversing the sign of a velocity."""
        if value > 0.0:
            return max(0.0, value - max_change)
        return min(0.0, value + max_change)

    def emergency_brake(self, state):
        """One *physically constrained* command, not an instant [0, 0].

        The fallback is best effort: when a new obstacle is already closer
        than the available braking distance, collision may be unavoidable.
        """
        return np.array([
            self._approach_zero(state[3], self.max_accel * self.dt),
            self._approach_zero(state[4], self.max_delta_yaw_rate * self.dt),
        ], dtype=float)

    def braking_trajectory(self, terminal_state):
        """Predict the stopping tail after a constant-control DWA rollout.

        Both linear and angular velocities decrease at their respective
        acceleration bounds. The last point has (approximately) zero speed.
        """
        state = terminal_state.copy()
        tail = []
        num_steps = max(
            int(math.ceil(abs(state[3]) / (self.max_accel * self.dt))),
            int(math.ceil(abs(state[4]) / (self.max_delta_yaw_rate * self.dt))),
        )
        for _ in range(num_steps):
            command = self.emergency_brake(state)
            state = self.motion(state, command)
            tail.append(state.copy())
        return np.asarray(tail, dtype=float).reshape(-1, 5)

    # ------------------------------------------------------------------
    def goal_cost(self, trajectory, goal):
        dx = goal[0] - trajectory[-1, 0]
        dy = goal[1] - trajectory[-1, 1]
        target_angle = math.atan2(dy, dx)
        angle_error = target_angle - trajectory[-1, 2]
        angle_error = math.atan2(math.sin(angle_error), math.cos(angle_error))
        return abs(angle_error) + 0.1 * math.hypot(dx, dy)

    @staticmethod
    def _prepare_path(global_path):
        """Tiền xử lý đoạn A* một lần/plan thay vì mỗi quỹ đạo ứng viên."""
        if global_path is None or len(global_path) < 2:
            return None
        A = global_path[:-1][None, :, :]
        AB = (global_path[1:] - global_path[:-1])[None, :, :]
        L2 = np.maximum((AB ** 2).sum(axis=2), 1e-12)
        return A, AB, L2

    @staticmethod
    def _path_cost_prepared(trajectory, prepared):
        if prepared is None:
            return 0.0
        A, AB, L2 = prepared
        P = trajectory[1:, :2][:, None, :]
        t = np.clip(((P - A) * AB).sum(axis=2) / L2, 0.0, 1.0)
        proj = A + t[..., None] * AB
        d = np.linalg.norm(P - proj, axis=2)
        return float(d.min(axis=1).mean())

    def path_cost(self, trajectory, global_path):
        """API cũ giữ nguyên để các module và test có thể gọi trực tiếp."""
        return self._path_cost_prepared(trajectory, self._prepare_path(global_path))

    def speed_cost(self, trajectory):
        return self.max_speed - trajectory[-1, 3]

    def obstacle_cost(self, trajectory, obs_pts, obs_vel, times):
        """Swept robot-vs-obstacle clearance across each time interval.

        Obstacle velocities are in world coordinates, inferred from LiDAR.
        Linear relative interpolation catches collisions *between* steps
        which point-only collision checks can miss.
        """
        if len(obs_pts) == 0:
            return 0.0
        if len(trajectory) < 2:
            return 0.0

        times = np.asarray(times, dtype=float)
        if len(times) != len(trajectory) - 1:
            raise ValueError("one timestamp is required per predicted step")
        timestamps = np.concatenate(([0.0], times))
        obs_at_t = obs_pts[None, :, :] + timestamps[:, None, None] * obs_vel[None, :, :]
        return self._swept_cost_prepared(trajectory, obs_at_t)

    def _swept_cost_prepared(self, trajectory, obs_at_t):
        """Giữ nguyên phép kiểm tra va chạm liên tục của Giai đoạn 2."""
        if obs_at_t.shape[1] == 0 or len(trajectory) < 2:
            return 0.0
        relative = trajectory[:, None, :2] - obs_at_t
        r0 = relative[:-1]
        change = relative[1:] - r0
        # 2D specialized: tránh 3 lần np.sum(..., axis=2) và tạo mảng
        # (K,N,2) trung gian cho closest, giữ nguyên công thức swept collision.
        dx = change[..., 0]
        dy = change[..., 1]
        norm2 = dx * dx + dy * dy
        dot = r0[..., 0] * dx + r0[..., 1] * dy
        fraction = np.clip(-dot / np.maximum(norm2, 1e-12), 0.0, 1.0)
        cx = r0[..., 0] + fraction * dx
        cy = r0[..., 1] + fraction * dy
        min_distance = float(np.sqrt(np.min(cx * cx + cy * cy)))

        if min_distance <= self.robot_radius + self.collision_margin:
            return float("inf")
        return 1.0 / (min_distance - self.robot_radius)

    # ------------------------------------------------------------------
    def _select_obstacles(self, state, pts, vel):
        if pts is None or len(pts) == 0:
            return np.empty((0, 2)), np.empty((0, 2))
        pts = np.asarray(pts, dtype=float)
        if vel is None:
            vel = np.zeros_like(pts)
        else:
            vel = np.asarray(vel, dtype=float)
        if vel.shape != pts.shape:
            raise ValueError("obstacle velocities must match obstacle point shape")

        # An obstacle can arrive from outside the old fixed 5 m radius.
        # Select candidates using upper bounds on robot/obstacle travel over
        # the DWA horizon *plus* the time needed to brake from maximum speed.
        brake_time = self.max_speed / self.max_accel
        horizon = self.predict_time + brake_time
        robot_reach = (self.max_speed * self.predict_time
                       + self.max_speed * brake_time / 2.0)
        relative_reach = (robot_reach + np.linalg.norm(vel, axis=1) * horizon
                          + self.robot_radius + self.collision_margin
                          + self.selection_margin)
        mask = np.linalg.norm(pts - state[:2], axis=1) <= relative_reach
        return pts[mask], vel[mask]

    def plan(self, state, goal, obstacle_points, obstacle_vel=None, global_path=None):
        dw = self.dynamic_window(state)
        obs_pts, obs_vel = self._select_obstacles(state, obstacle_points, obstacle_vel)

        steps = int(self.predict_time / self.dt)

        best_cost = float("inf")
        best_control = None
        best_trajectory = None

        v_samples = np.arange(dw[0], dw[1] + self.v_resolution * 0.5, self.v_resolution)
        omega_samples = np.arange(dw[2], dw[3] + self.yaw_rate_resolution * 0.5,
                                  self.yaw_rate_resolution)

        # Những dữ liệu sau chỉ phụ thuộc vào đường A* và frame LiDAR,
        # không phụ thuộc vào (v, omega) của quỹ đạo ứng viên.
        prepared_path = self._prepare_path(global_path)
        max_v = max(abs(self.min_speed), abs(self.max_speed))
        max_tail = max(
            math.ceil(max_v / (self.max_accel * self.dt)),
            math.ceil(self.max_yaw_rate / (self.max_delta_yaw_rate * self.dt))
        )
        # +3 bảo vệ khi float v=1.5000000000000002 làm ceil thêm 1 bước.
        timestamps = np.arange(steps + max_tail + 4) * self.dt
        obs_at_t = (obs_pts[None, :, :] + timestamps[:, None, None] * obs_vel[None, :, :]
                    if len(obs_pts) else np.empty((len(timestamps), 0, 2)))

        for v in v_samples:
            for omega in omega_samples:
                control = [v, omega]
                traj = self.predict_trajectory(state, control)

                # The constant-speed prediction is only feasible if the
                # robot can also brake to rest beyond its 2 s horizon.
                tail = self.braking_trajectory(traj[-1])
                safe_path = np.vstack((traj, tail)) if len(tail) else traj
                # Vị trí vật cản theo thời gian được tính một lần ở trên.
                ob_cost = self._swept_cost_prepared(
                    safe_path, obs_at_t[:len(safe_path)]
                )
                if not np.isfinite(ob_cost):
                    continue

                cost = (
                    self.to_goal_gain * self.goal_cost(traj, goal)
                    + self.path_gain * self._path_cost_prepared(traj, prepared_path)
                    + self.speed_gain * self.speed_cost(traj)
                    + self.obstacle_gain * ob_cost
                )

                if cost < best_cost:
                    best_cost = cost
                    best_control = control
                    best_trajectory = traj

        return best_control, best_trajectory, best_cost