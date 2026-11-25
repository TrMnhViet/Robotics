
import numpy as np


class Robot:
    def __init__(self, x=1.0, y=1.0, yaw=0.0):
        # State: [x, y, yaw, v, omega]
        self.state = np.array([x, y, yaw, 0.0, 0.0], dtype=float)

    def motion(self, control, dt):
        v, omega = control
        x = self.state.copy()

        x[2] += omega * dt
        x[2] = (x[2] + np.pi) % (2 * np.pi) - np.pi

        x[0] += v * np.cos(x[2]) * dt
        x[1] += v * np.sin(x[2]) * dt

        x[3] = v
        x[4] = omega

        self.state = x
        return self.state.copy()

    def get_state(self):
        return self.state.copy()

    def reset(self, x=1.0, y=1.0, yaw=0.0):
        self.state = np.array([x, y, yaw, 0.0, 0.0], dtype=float)
