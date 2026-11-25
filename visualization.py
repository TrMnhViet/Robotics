
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Circle, Rectangle
from matplotlib.collections import LineCollection


class Visualizer:
    def __init__(self, env, goal):
        self.env = env
        self.goal = goal

        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.ax.set_xlim(0, env.width)
        self.ax.set_ylim(0, env.height)
        self.ax.set_aspect("equal")
        self.ax.grid(True)
        self.ax.set_title("DWA + LiDAR 360 Simulation")

        # Draw obstacles
        for obs in env.circles:
            self.ax.add_patch(
                Circle((obs.x, obs.y), obs.radius, color="gray")
            )

        for obs in env.rectangles:
            self.ax.add_patch(
                Rectangle(
                    (obs.xmin, obs.ymin),
                    obs.xmax - obs.xmin,
                    obs.ymax - obs.ymin,
                    color="gray"
                )
            )

        # Moving obstacles: create patches once, then update each frame.
        self.dynamic_patches = []
        self.dynamic_labels = []
        for index, obs in enumerate(env.dynamic_obstacles, start=1):
            patch = Circle(
                (obs.x, obs.y), obs.radius,
                facecolor="darkorange", edgecolor="saddlebrown",
                linewidth=1.4, alpha=0.9, zorder=5,
                label="Dynamic obstacle" if index == 1 else None
            )
            self.ax.add_patch(patch)
            self.dynamic_patches.append(patch)
            label = self.ax.text(
                obs.x, obs.y, f"D{index}",
                ha="center", va="center", color="white",
                fontweight="bold", fontsize=9, zorder=6
            )
            self.dynamic_labels.append(label)

        # Goal
        self.ax.plot(goal[0], goal[1], "r*", markersize=14, label="Goal")

        # Robot
        self.robot_patch = Circle(
            (0, 0), 0.3,
            color="blue",
            alpha=0.8
        )
        self.ax.add_patch(self.robot_patch)

        self.heading_line, = self.ax.plot([], [], "b-", linewidth=2)

        # LiDAR
        self.lidar_points = self.ax.scatter(
            [], [], c="red", s=8, label="LiDAR points"
        )

        self.lidar_lines = LineCollection(
            [], colors="green", linewidths=0.4, alpha=0.4
        )
        self.ax.add_collection(self.lidar_lines)

        # Trajectories
        self.predict_line, = self.ax.plot(
            [], [], "b--", linewidth=1.5, label="DWA prediction"
        )

        self.actual_line, = self.ax.plot(
            [], [], "orange", linewidth=1.5, label="Robot path"
        )

        # Một Line2D duy nhất cho A*: replan chỉ cập nhật dữ liệu,
        # không tạo chồng hàng chục đường và legend trên bản đồ.
        self.global_path_line, = self.ax.plot(
            [], [], "c--", linewidth=1.5, label="A* Global Path"
        )
        self.ax.legend(loc="upper left")
        self.mode_label = self.ax.text(
            0.98, 0.98, "Mode: NORMAL", transform=self.ax.transAxes,
            ha="right", va="top", fontsize=11, fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="gray"),
            zorder=10
        )

        # Matplotlib blitting: vẽ nền bản đồ một lần; mỗi frame chỉ vẽ
        # robot, vật cản động, tia LiDAR và đường đi đang thay đổi.
        # Nếu backend không hỗ trợ, tự động dùng draw_idle như GĐ3.
        self._animated = (
            self.dynamic_patches + self.dynamic_labels + [
                self.robot_patch, self.heading_line, self.lidar_points,
                self.lidar_lines, self.predict_line, self.actual_line,
                self.mode_label,
            ]
        )
        self._use_blit = bool(self.fig.canvas.supports_blit)
        self._background = None
        if self._use_blit:
            for artist in self._animated:
                artist.set_animated(True)
            self.fig.canvas.mpl_connect("draw_event", self._on_full_draw)
            self.fig.canvas.draw()

    def _on_full_draw(self, event):
        if self._use_blit:
            self._background = self.fig.canvas.copy_from_bbox(self.ax.bbox)

    def update(self, state, scan, trajectory, path):
        x, y, yaw = state[:3]

        # Keep the drawn circles synchronized with Environment.step(DT).
        for obs, patch, label in zip(
            self.env.dynamic_obstacles,
            self.dynamic_patches,
            self.dynamic_labels
        ):
            patch.center = (obs.x, obs.y)
            label.set_position((obs.x, obs.y))

        # Update robot position
        self.robot_patch.center = (x, y)

        # Robot heading
        heading_x = x + 0.6 * np.cos(yaw)
        heading_y = y + 0.6 * np.sin(yaw)

        self.heading_line.set_data(
            [x, heading_x],
            [y, heading_y]
        )

        # Update LiDAR points
        points = scan["points"]

        self.lidar_points.set_offsets(
            points if len(points) > 0 else np.empty((0, 2))
        )

        # Update LiDAR rays
        if len(points) > 0:
            origins = np.repeat(
                [[x, y]], len(points), axis=0
            )
            segments = np.stack((origins, points), axis=1)
            self.lidar_lines.set_segments(segments)
        else:
            self.lidar_lines.set_segments([])

        # Predicted trajectory
        if trajectory is not None:
            self.predict_line.set_data(
                trajectory[:, 0],
                trajectory[:, 1]
            )
        else:
            self.predict_line.set_data([], [])

        # Actual robot path
        if len(path) > 0:
            path_array = np.array(path)
            self.actual_line.set_data(
                path_array[:, 0],
                path_array[:, 1]
            )

        if self._use_blit and self._background is not None:
            canvas = self.fig.canvas
            canvas.restore_region(self._background)
            for artist in self._animated:
                self.ax.draw_artist(artist)
            canvas.blit(self.ax.bbox)
            canvas.flush_events()
        else:
            self.fig.canvas.draw_idle()
        plt.pause(0.001)

    def show(self):
        # Sau mô phỏng, đảm bảo figure vẽ đầy đủ tất cả các nghệ sĩ động
        # kể cả khi cửa sổ được resize/minimize sau khung cuối.
        if self._use_blit:
            for artist in self._animated:
                artist.set_animated(False)
            self._use_blit = False
            self.fig.canvas.draw_idle()
        plt.show()

    def set_mode(self, mode):
        self.mode_label.set_text(f"Mode: {mode}")
        self.mode_label.set_color(
            "darkred" if mode == "WAIT" else
            "darkorange" if mode == "YIELD" else
            "darkgreen" if mode == "RESUME" else "black"
        )

    def set_global_path(self, path):
        self.global_path_line.set_data(path[:, 0], path[:, 1])
        if self._use_blit:
            # Đường A* là thành phần nền; chỉ dựng lại nền khi replan.
            self.fig.canvas.draw()
