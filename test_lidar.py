
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

from environment import Environment
from robot import Robot
from lidar import LiDAR


env = Environment()
robot = Robot(x=1.0, y=1.0, yaw=0.0)
lidar = LiDAR()

state = robot.get_state()
scan = lidar.scan(state, env)

points = scan["points"]

fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xlim(0, env.width)
ax.set_ylim(0, env.height)
ax.set_aspect("equal")
ax.grid(True)

# Draw circular obstacles
for obs in env.circles:
    ax.add_patch(Circle((obs.x, obs.y), obs.radius, color="gray"))

# Draw rectangular obstacles
for obs in env.rectangles:
    ax.add_patch(Rectangle(
        (obs.xmin, obs.ymin),
        obs.xmax - obs.xmin,
        obs.ymax - obs.ymin,
        color="gray"
    ))

# Draw robot
ax.plot(state[0], state[1], "bo", markersize=10)

# Draw LiDAR rays
for point in points:
    ax.plot(
        [state[0], point[0]],
        [state[1], point[1]],
        color="green",
        linewidth=0.4,
        alpha=0.5
    )

# Draw detected points
if len(points) > 0:
    ax.scatter(points[:, 0], points[:, 1], c="red", s=8)

ax.set_title("LiDAR 360 Degree Simulation")

print("Number of rays:", lidar.num_rays)
print("Detected points:", len(points))
print("Ranges shape:", scan["ranges"].shape)
print("Points shape:", points.shape)

plt.show()
