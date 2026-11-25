
import numpy as np

from robot import Robot
from environment import Environment
from lidar import LiDAR
from dwa import DWA


env = Environment()
robot = Robot(x=1.0, y=1.0, yaw=0.0)
lidar = LiDAR()
dwa = DWA()

goal = np.array([18.0, 18.0])

# LiDAR scan
state = robot.get_state()
scan = lidar.scan(state, env)
obstacle_points = scan["points"]

# DWA planning
control, trajectory, cost = dwa.plan(
    state,
    goal,
    obstacle_points
)

print("Robot state:", state)
print("Goal:", goal)
print("Detected obstacles:", len(obstacle_points))
print("Selected control:", control)
print("Best cost:", cost)

if trajectory is not None:
    print("Predicted trajectory shape:", trajectory.shape)
    print("Predicted final position:", trajectory[-1, :2])
else:
    print("No valid trajectory found!")
