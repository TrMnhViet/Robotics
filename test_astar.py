
import sys
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.patches import Circle, Rectangle

from environment import Environment
from astar import AStarPlanner


# Initialize environment
env = Environment()

start = np.array([1.0, 1.0])
goal = np.array([18.0, 18.0])

# Initialize A*
planner = AStarPlanner(env)

# Find path
path = planner.plan(start, goal)

if path is None:
    print("No path found!")
    sys.exit(1)

# Calculate path length
distances = np.linalg.norm(
    np.diff(path, axis=0),
    axis=1
)
path_length = np.sum(distances)

print("A* planning successful!")
print("Number of waypoints:", len(path))
print("Path length:", round(path_length, 3), "m")
print("Start:", path[0])
print("Goal:", path[-1])

# Draw map
fig, ax = plt.subplots(figsize=(10, 10))

ax.set_xlim(0, env.width)
ax.set_ylim(0, env.height)
ax.set_aspect("equal")
ax.grid(True)

# Draw inflated occupancy grid
ax.imshow(
    planner.grid,
    origin="lower",
    extent=[0, env.width, 0, env.height],
    cmap="Greys",
    alpha=0.25,
    interpolation="nearest"
)

# Draw original circular obstacles
for obs in env.circles:
    ax.add_patch(
        Circle(
            (obs.x, obs.y),
            obs.radius,
            color="gray"
        )
    )

# Draw original rectangular obstacles
for obs in env.rectangles:
    ax.add_patch(
        Rectangle(
            (obs.xmin, obs.ymin),
            obs.xmax - obs.xmin,
            obs.ymax - obs.ymin,
            color="gray"
        )
    )

# Draw path
ax.plot(
    path[:, 0],
    path[:, 1],
    "b-",
    linewidth=2,
    label="A* Path"
)

ax.scatter(
    path[:, 0],
    path[:, 1],
    c="blue",
    s=8
)

# Start and goal
ax.plot(*start, "go", markersize=10, label="Start")
ax.plot(*goal, "r*", markersize=14, label="Goal")

ax.set_title(
    f"A* Global Path Planning - Length: {path_length:.2f} m"
)
ax.legend()

plt.show()
