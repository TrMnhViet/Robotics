"""Optional Stage-3 demo: one circle crosses a straight robot route.

Run `python demo_crossing.py` to watch YIELD / WAIT / RESUME.
Running `python main.py` keeps the original complex obstacle map unchanged.
"""
import main
from environment import Environment, DynamicCircleObstacle


class CrossingDemoEnvironment(Environment):
    def __init__(self):
        super().__init__()
        self.circles = []
        self.rectangles = []
        self.dynamic_obstacles = [
            DynamicCircleObstacle(6.5, 5.0, 0.42, 0.0, 0.65),
        ]


def run():
    main.Environment = CrossingDemoEnvironment
    main.START_X, main.START_Y, main.START_YAW = 1.0, 10.0, 0.0
    main.GOAL_X, main.GOAL_Y = 18.0, 10.0
    main.main()


if __name__ == "__main__":
    run()
