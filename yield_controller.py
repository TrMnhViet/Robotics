"""Stage 3: yield / wait / resume overlay on top of A* + DWA.

A crossing circular obstacle is evaluated against the *nominal global path*.
When DWA is already slowing for that obstacle, the controller records YIELD
but preserves DWA's safe control. If a more decisive stop is appropriate,
the controller only overrides DWA when there is enough time to stop AND the
physically constrained braking path is safe against every LiDAR return.

The overlay does not replace DWA's collision checking. The chosen braking
path is rechecked every step, and the controller defers to DWA if a safe stop
is no longer available. The model is a simulation, not a safety guarantee.
"""

from dataclasses import dataclass
import math
import numpy as np
from config import DT, ROBOT_RADIUS


@dataclass(frozen=True)
class CrossingRisk:
    track_id: int
    time_to_conflict: float
    min_clearance: float


class YieldController:
    NORMAL = "NORMAL"
    YIELD = "YIELD"
    WAIT = "WAIT"
    RESUME = "RESUME"

    def __init__(self, dt=DT, horizon=3.0, margin=0.25,
                 lateral_speed=0.25, clear_frames=6, confirm_frames=2,
                 resume_frames=8, hold_time=1.0,
                 comfort_clearance=0.75, reaction_time=0.2):
        self.dt = dt
        self.horizon = horizon
        self.margin = margin
        self.lateral_speed = lateral_speed
        self.clear_frames = clear_frames
        self.confirm_frames = confirm_frames
        self.resume_frames = resume_frames
        self.hold_time = hold_time
        self.comfort_clearance = comfort_clearance
        self.reaction_time = reaction_time
        self.mode = self.NORMAL
        self.clear_count = 0
        self.risk_count = 0
        self.resume_count = 0
        self.last_risk = None
        self.entries = 0
        self.wait_steps = 0
        self.resume_events = 0
        self.forced_brake = False

    def _nominal_path(self, state, path, max_speed):
        """Robot centers and local path tangents along a sampled global route."""
        if path is None or len(path) < 2:
            return None

        route = np.asarray(path, dtype=float)
        start = np.asarray(state[:2], dtype=float)
        nearest = int(np.argmin(np.linalg.norm(route - start, axis=1)))
        # Start from the current position (not from an old waypoint behind us).
        forward = np.vstack((start, route[nearest + 1:]))
        if len(forward) < 2:
            forward = np.vstack((start, route[-1]))

        delta = np.diff(forward, axis=0)
        length = np.linalg.norm(delta, axis=1)
        valid = length > 1e-8
        if not np.any(valid):
            return None
        # Drop zero-length segments without changing the direction of travel.
        forward = np.vstack((forward[0], forward[1:][valid]))
        delta = np.diff(forward, axis=0)
        length = np.linalg.norm(delta, axis=1)
        cumulative = np.r_[0., np.cumsum(length)]
        tangent = delta / length[:, None]

        # In WAIT, use a modest restart speed rather than speed=0, otherwise
        # it would incorrectly conclude that the path is always clear.
        speed = max(0.8, min(max_speed, max(0., float(state[3])) + 0.2))
        times = np.arange(0., self.horizon + self.dt * 0.5, self.dt)
        distance = np.minimum(speed * times, cumulative[-1])
        segment = np.clip(np.searchsorted(cumulative, distance, side="right") - 1,
                          0, len(length) - 1)
        factor = (distance - cumulative[segment]) / length[segment]
        positions = forward[segment] + factor[:, None] * delta[segment]
        return times, positions, tangent[segment]

    def crossing_risk(self, state, path, tracks, max_speed):
        sampled = self._nominal_path(state, path, max_speed)
        if sampled is None:
            return None
        times, robot_positions, tangents = sampled
        best = None

        for track in tracks:
            # dynamic_tracks() already enforces min_hits, size and speed.
            # These extra checks also make direct unit testing robust.
            if getattr(track, "missed", 0) or getattr(track, "hits", 3) < 3:
                continue
            velocity = np.asarray(track.vel, dtype=float)
            lateral = np.abs(tangents[:, 0] * velocity[1]
                             - tangents[:, 1] * velocity[0])
            if float(lateral.max()) < self.lateral_speed:
                continue

            obstacle_positions = np.asarray(track.pos, dtype=float)[None, :] + times[:, None] * velocity
            distances = np.linalg.norm(robot_positions - obstacle_positions, axis=1)
            radius = 0.5 * float(track.size) + ROBOT_RADIUS + self.margin
            hit = (times >= self.dt) & (distances <= radius) & (lateral >= self.lateral_speed)
            if not np.any(hit):
                continue
            first = int(np.flatnonzero(hit)[0])
            risk = CrossingRisk(int(track.id), float(times[first]),
                                float(distances.min() - radius))
            if best is None or risk.time_to_conflict < best.time_to_conflict:
                best = risk
        return best

    def _close_to_dwa_trajectory(self, state, risk, tracks, command, dwa):
        """Yield only if DWA's current collision-free route passes too close.

        A* can cross an obstacle's expected path while DWA already has a
        comfortable detour. Forcing a full stop in that situation can cause
        an avoidable local minimum when restarting.
        """
        track = next((tr for tr in tracks if tr.id == risk.track_id), None)
        if track is None:
            return False
        trajectory = dwa.predict_trajectory(state, command)
        times = np.arange(len(trajectory)) * dwa.dt
        obstacle = np.asarray(track.pos)[None, :] + times[:, None] * np.asarray(track.vel)
        center_dist = np.linalg.norm(trajectory[:, :2] - obstacle, axis=1)
        free_gap = float(center_dist.min() - 0.5 * track.size - ROBOT_RADIUS)
        return free_gap < self.comfort_clearance

    def _brake_and_hold_safe(self, state, obstacle_points, obstacle_velocities, dwa):
        """Before overriding DWA, verify the brake-and-wait path is safe."""
        points, velocities = dwa._select_obstacles(state, obstacle_points, obstacle_velocities)
        if len(points) == 0:
            return True

        tail = dwa.braking_trajectory(state)
        final = tail[-1] if len(tail) else state
        hold_steps = max(1, math.ceil(self.hold_time / self.dt))
        hold = np.repeat(final[None, :], hold_steps, axis=0)
        projected = np.vstack((state, tail, hold))
        timestamps = np.arange(1, len(projected)) * self.dt
        return bool(np.isfinite(dwa.obstacle_cost(projected, points, velocities, timestamps)))

    def step(self, state, global_path, tracks, dwa_control, obstacle_points,
             obstacle_velocities, dwa):
        """Return (control, mode_change_message or None).

        Yielding is cooperative: if DWA is already slowing near a predicted
        crossing, record YIELD without interrupting its safe trajectory.
        Physically constrained braking is requested only if the DWA trajectory
        comes too close AND there is still enough time to stop.
        """
        old_mode = self.mode
        raw_risk = self.crossing_risk(state, global_path, tracks, dwa.max_speed)
        risk = raw_risk
        passive = False
        forced = False

        if raw_risk is not None and self.mode in (self.NORMAL, self.RESUME):
            time_to_stop = abs(float(state[3])) / dwa.max_accel
            # DWA is slowing by itself because of a crossing. Do not replace
            # that collision-free steering command with arbitrary full braking.
            passive = (raw_risk.time_to_conflict <= 2.5
                       and abs(float(dwa_control[0])) <= 0.4
                       and abs(float(state[3])) <= 0.6)
            forced = (raw_risk.time_to_conflict >= time_to_stop + self.reaction_time
                      and self._close_to_dwa_trajectory(
                          state, raw_risk, tracks, dwa_control, dwa))
            if not passive and not forced:
                risk = None
        self.last_risk = risk
        self.risk_count = self.risk_count + 1 if risk is not None else 0

        stop_is_safe = None
        if (forced and self.risk_count >= self.confirm_frames
                or self.mode in (self.YIELD, self.WAIT)):
            stop_is_safe = self._brake_and_hold_safe(
                state, obstacle_points, obstacle_velocities, dwa
            )

        if self.mode in (self.NORMAL, self.RESUME):
            if (self.risk_count >= self.confirm_frames
                    and (passive or forced and stop_is_safe)):
                self.mode = self.YIELD
                self.forced_brake = bool(forced and stop_is_safe)
                self.clear_count = 0
                self.resume_count = 0
                self.entries += 1
            elif self.mode == self.RESUME:
                self.resume_count += 1
                if self.resume_count >= self.resume_frames:
                    self.mode = self.NORMAL

        # If stopping has become unsafe, DWA must be allowed to steer away.
        if (self.mode in (self.YIELD, self.WAIT)
                and stop_is_safe is False and self.forced_brake):
            self.mode = self.RESUME
            self.forced_brake = False
            self.clear_count = 0
            self.resume_count = 0

        if self.mode == self.YIELD:
            command = (dwa.emergency_brake(state) if self.forced_brake
                       else np.asarray(dwa_control, dtype=float))
            if risk is None:
                self.clear_count += 1
            else:
                self.clear_count = 0
            if (abs(state[3]) <= 0.06 and abs(command[0]) <= 0.06):
                self.mode = self.WAIT
                self.clear_count = 0
            elif (not self.forced_brake and self.clear_count >= self.clear_frames
                  and abs(command[0]) > 0.45):
                self.mode = self.RESUME
                self.resume_events += 1
                self.resume_count = 0
        elif self.mode == self.WAIT:
            self.wait_steps += 1
            if stop_is_safe is False:
                self.mode = self.RESUME
                self.resume_count = 0
                self.resume_events += 1
                command = np.asarray(dwa_control, dtype=float)
            else:
                command = dwa.emergency_brake(state)
                self.clear_count = self.clear_count + 1 if risk is None else 0
                if self.clear_count >= self.clear_frames:
                    self.mode = self.RESUME
                    self.forced_brake = False
                    self.resume_count = 0
                    self.resume_events += 1
                    command = np.asarray(dwa_control, dtype=float)
        else:
            command = np.asarray(dwa_control, dtype=float)

        if self.mode != old_mode:
            info = f"{old_mode} -> {self.mode}"
            if risk is not None:
                info += f" (track ID {risk.track_id}, TTC={risk.time_to_conflict:.1f}s)"
            return command, info
        return command, None
