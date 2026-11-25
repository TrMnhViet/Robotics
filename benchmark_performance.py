"""Do toc do mo phong khong mo cua so matplotlib.

Chay:
    python benchmark_performance.py
    python benchmark_performance.py --scenario crossing
    python benchmark_performance.py --scenario static

Thoi gian la wall-clock tren may dang chay, KHONG phai FPS GUI.
"""
import argparse
import time

import main


class HeadlessVisualizer:
    def __init__(self, env, goal):
        pass

    def set_global_path(self, path):
        pass

    def update(self, state, scan, trajectory, path):
        pass

    def set_mode(self, mode):
        pass

    def show(self):
        pass


def main_benchmark(scenario):
    main.Visualizer = HeadlessVisualizer
    if scenario == "crossing":
        from environment import Environment, DynamicCircleObstacle

        class CrossingDemoEnvironment(Environment):
            def __init__(self):
                super().__init__()
                self.circles = []
                self.rectangles = []
                self.dynamic_obstacles = [
                    DynamicCircleObstacle(6.5, 5.0, 0.42, 0.0, 0.65),
                ]

        main.Environment = CrossingDemoEnvironment
        main.START_X, main.START_Y, main.START_YAW = 1.0, 10.0, 0.0
        main.GOAL_X, main.GOAL_Y = 18.0, 10.0
    elif scenario == "static":
        from environment import Environment

        class StaticEnvironment(Environment):
            def __init__(self):
                super().__init__()
                self.dynamic_obstacles = []

        main.Environment = StaticEnvironment

    start = time.perf_counter()
    main.main()
    print(f"Elapsed (no GUI, no profiler): {time.perf_counter()-start:.3f} s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=("nominal", "crossing", "static"),
                        default="nominal")
    main_benchmark(parser.parse_args().scenario)
