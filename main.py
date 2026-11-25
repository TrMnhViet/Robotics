
import numpy as np

from config import (
    START_X, START_Y, START_YAW,
    GOAL_X, GOAL_Y,
    GOAL_TOLERANCE,
    MAX_SIMULATION_STEPS,
    DT, ROBOT_RADIUS
)

from robot import Robot
from environment import Environment
from lidar import LiDAR
from astar import AStarPlanner
from occupancy_grid import OccupancyGrid
from pure_pursuit import PurePursuit
from dwa import DWA
from obstacle_tracker import ObstacleTracker
from yield_controller import YieldController
from visualization import Visualizer

REPLAN_COOLDOWN = 10
BLOCK_CHECK_POINTS = 40


def is_path_blocked(path, robot_position, planner, horizon=BLOCK_CHECK_POINTS):
    if path is None or len(path) == 0:
        return True

    distances = np.linalg.norm(path - robot_position, axis=1)
    nearest_index = int(np.argmin(distances))

    for point in path[nearest_index:nearest_index + horizon]:
        col, row = planner.world_to_grid(point[0], point[1])
        if not planner.is_valid(col, row):
            return True

    return False


def main():
    env = Environment()
    robot = Robot(START_X, START_Y, START_YAW)
    lidar = LiDAR()
    occupancy_grid = OccupancyGrid()
    tracker = ObstacleTracker()
    dwa = DWA()
    yield_controller = YieldController()
    # Pure Pursuit chỉ còn dùng để chọn điểm lookahead trên đường A*
    pure_pursuit = PurePursuit(lookahead_distance=1.5)

    goal = np.array([GOAL_X, GOAL_Y])
    visualizer = Visualizer(env, goal)

    global_path = None
    actual_path = []
    success = False
    replan_count = 0
    collision_steps = 0
    emergency_brake_steps = 0
    steps_since_replan = REPLAN_COOLDOWN

    print("Starting A* + DWA + LiDAR + dynamic obstacle tracking...")

    for step in range(MAX_SIMULATION_STEPS):

        env.step(DT)

        state = robot.get_state()

        if np.linalg.norm(state[:2] - goal) <= GOAL_TOLERANCE:
            success = True
            print("Goal reached!")
            break

        if env.in_collision(state[0], state[1], ROBOT_RADIUS):
            collision_steps += 1


        scan = lidar.scan(state, env)
        occupancy_grid.update(state, scan)


        points, velocities = tracker.update(scan["points"], DT)


        steps_since_replan += 1
        need_replan = False
        if global_path is None or steps_since_replan >= REPLAN_COOLDOWN:
            astar = AStarPlanner(occupancy_grid)
            need_replan = (global_path is None or
                           is_path_blocked(global_path, state[:2], astar))

        if need_replan:
            print(f"Step {step}: Replanning path...")
            new_path = astar.plan(state[:2], goal)
            steps_since_replan = 0

            if new_path is None:
                if global_path is None:
                    print("A*: No path found with the current map.")
                    break
                print("A*: replan failed, keeping old path.")
            else:
                global_path = new_path
                pure_pursuit.reset()
                replan_count += 1
                visualizer.set_global_path(global_path)

        # 4. Local planner (DWA) bám theo điểm lookahead, né vật cản động
        target = pure_pursuit.find_target(state[:2], global_path)
        control, trajectory, cost = dwa.plan(
            state, target, points, velocities, global_path
        )

        if control is None or not np.isfinite(cost):
            # No safe DWA trajectory: brake at the physical acceleration limit.
            # Instant [0, 0] violates MAX_ACCEL when already moving quickly.
            control = dwa.emergency_brake(state)
            trajectory = np.vstack((state, dwa.motion(state, control)))
            emergency_brake_steps += 1
            steps_since_replan = REPLAN_COOLDOWN

        # Stage 3: yield to a CONFIRMED moving obstacle crossing the
        # intended A* path, but only if the stopping trajectory is safe.
        # Otherwise keep the safe control chosen by DWA (Stage 2).
        original_control = np.asarray(control, dtype=float)
        control, change = yield_controller.step(
            state, global_path, tracker.dynamic_tracks(), original_control,
            points, velocities, dwa
        )
        if change is not None:
            print(f"Step {step}: Yield controller {change}")
            visualizer.set_mode(yield_controller.mode)

        # Draw the actual commanded trajectory instead of the discarded DWA
        # prediction whenever the controller orders a stop/wait.
        if not np.allclose(control, original_control):
            stop_tail = dwa.braking_trajectory(state)
            trajectory = np.vstack((state, stop_tail)) if len(stop_tail) else np.array([state.copy()])

        # 5. Visualization
        actual_path.append(state[:2].copy())
        visualizer.update(state, scan, trajectory, actual_path)

        # 6. Move robot
        robot.motion(control, DT)

        if step % 20 == 0:
            print(
                f"Step: {step:4d} | "
                f"Pos: ({state[0]:.2f}, {state[1]:.2f}) | "
                f"v: {control[0]:.2f} | "
                f"omega: {np.degrees(control[1]):.1f} deg/s | "
                f"Dyn tracks: {len(tracker.dynamic_tracks())} | "
                f"Replans: {replan_count} | "
                f"Mode: {yield_controller.mode}"
            )

    final_state = robot.get_state()

    print("\nSimulation finished.")
    print("Success:", success)
    print("Replanning count:", replan_count)
    print("Collision steps:", collision_steps)
    print("Emergency brake steps:", emergency_brake_steps)
    print("Yield events:", yield_controller.entries)
    print("Wait steps:", yield_controller.wait_steps)
    print("Resume events:", yield_controller.resume_events)
    print(f"Final position: ({final_state[0]:.2f}, {final_state[1]:.2f})")
    print(f"Distance to goal: {np.linalg.norm(final_state[:2] - goal):.3f} m")

    # Closing the final Matplotlib window with Ctrl+C is not a planner error.
    try:
        visualizer.show()
    except KeyboardInterrupt:
        print("Visualization interrupted by user (Ctrl+C).")


if __name__ == "__main__":
    main()
