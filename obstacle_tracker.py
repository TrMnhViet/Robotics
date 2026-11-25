"""LiDAR-based tracking of circular moving obstacles.

Only moving obstacles that form reliable circular LiDAR clusters are assigned
non-zero velocities. All other LiDAR points remain stationary obstacles for
DWA, so an uncertain classification never removes an obstacle from avoidance.
"""

import math

import numpy as np


class Track:
    _next_id = 0

    def __init__(self, pos, radius, timestamp):
        self.id = Track._next_id
        Track._next_id += 1
        self.pos = np.asarray(pos, dtype=float).copy()
        self.vel = np.zeros(2, dtype=float)
        self.size = 2.0 * radius
        self.hits = 1
        self.missed = 0
        self.last_seen = timestamp
        self.history = [(timestamp, self.pos.copy())]

    def update(self, pos, radius, timestamp, alpha):
        pos = np.asarray(pos, dtype=float)
        self.history.append((timestamp, pos.copy()))
        self.history = self.history[-5:]
        first_time, first_pos = self.history[0]
        elapsed = timestamp - first_time
        if elapsed > 1e-8:
            measured_vel = (pos - first_pos) / elapsed
            self.vel = alpha * measured_vel + (1.0 - alpha) * self.vel
        self.pos = pos.copy()
        self.size = 2.0 * radius
        self.last_seen = timestamp
        self.hits += 1
        self.missed = 0


class ObstacleTracker:
    """Track circles by fitted *world-space centers*, not LiDAR surface means.

    update(points, dt) -> (points[N,2], velocities[N,2]) remains unchanged.
    Each measured point receives its cluster's estimated velocity only after
    enough consistent observations. Non-circular / uncertain clusters get 0.
    """

    def __init__(self, cluster_gap=0.4, gate=0.5, alpha=0.4,
                 static_speed=0.3, min_hits=3, max_missed=3,
                 max_dynamic_size=2.0, min_circle_points=5,
                 circle_residual=0.025, min_visible_arc=0.5):
        self.cluster_gap = cluster_gap
        self.gate = gate
        self.alpha = alpha
        self.static_speed = static_speed
        self.min_hits = min_hits
        self.max_missed = max_missed
        self.max_dynamic_size = max_dynamic_size
        self.min_circle_points = min_circle_points
        self.circle_residual = circle_residual
        self.min_visible_arc = min_visible_arc
        self.tracks = []
        self._time = 0.0

    def _cluster(self, points):
        if len(points) == 0:
            return []
        gaps = np.linalg.norm(np.diff(points, axis=0), axis=1) > self.cluster_gap
        clusters = list(np.split(np.arange(len(points)), np.flatnonzero(gaps) + 1))

        # The first and last valid rays may belong to the same 360-degree arc.
        if (len(clusters) > 1
                and np.linalg.norm(points[0] - points[-1]) <= self.cluster_gap):
            clusters[0] = np.concatenate((clusters[-1], clusters[0]))
            clusters.pop()
        return clusters

    def _circle_detection(self, pts):
        """Return center/radius only for a sufficiently visible circular arc.

        Algebraic least squares in centered coordinates works well for the
        exact synthetic LiDAR rays used by this project. Reject wall segments
        and rectangle corners by radius, residual and visible angular arc.
        """
        if len(pts) < self.min_circle_points:
            return None

        origin = pts.mean(axis=0)
        xy = pts - origin
        a = np.column_stack((2.0 * xy, np.ones(len(pts))))
        try:
            params, _, rank, _ = np.linalg.lstsq(a, np.sum(xy * xy, axis=1), rcond=None)
        except np.linalg.LinAlgError:
            return None
        if rank < 3:
            return None

        radius_sq = params[2] + np.dot(params[:2], params[:2])
        if radius_sq <= 0.0:
            return None
        radius = math.sqrt(radius_sq)
        if not (0.2 <= radius <= self.max_dynamic_size / 2.0):
            return None

        center = origin + params[:2]
        radial_distance = np.linalg.norm(pts - center, axis=1)
        residual = np.sqrt(np.mean((radial_distance - radius) ** 2))
        if residual > self.circle_residual:
            return None

        angles = np.sort(np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0]))
        gaps = np.diff(np.concatenate((angles, angles[:1] + 2.0 * np.pi)))
        visible_arc = 2.0 * np.pi - float(gaps.max())
        if visible_arc < self.min_visible_arc:
            return None

        return center, radius

    def _is_dynamic(self, track):
        return (track.missed == 0
                and track.hits >= self.min_hits
                and track.size <= self.max_dynamic_size
                and np.linalg.norm(track.vel) >= self.static_speed)

    def update(self, points, dt):
        if dt <= 0:
            raise ValueError("dt must be positive")
        self._time += dt
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        velocities = np.zeros_like(points)

        detections = []
        for indices in self._cluster(points):
            fitted = self._circle_detection(points[indices])
            if fitted is not None:
                center, radius = fitted
                detections.append((indices, center, radius))

        # Match by predicted circle *center*, not by visible surface centroid.
        candidates = []
        for ti, tr in enumerate(self.tracks):
            predicted = tr.pos + tr.vel * (self._time - tr.last_seen)
            for di, (_, center, radius) in enumerate(detections):
                if abs(radius - tr.size / 2.0) > 0.2:
                    continue
                distance = float(np.linalg.norm(center - predicted))
                if distance <= self.gate:
                    candidates.append((distance, ti, di))
        candidates.sort()

        matched_tracks, matched_detections = set(), set()
        associated = {}
        for _, ti, di in candidates:
            if ti in matched_tracks or di in matched_detections:
                continue
            matched_tracks.add(ti)
            matched_detections.add(di)
            tr = self.tracks[ti]
            _, center, radius = detections[di]
            tr.update(center, radius, self._time, self.alpha)
            associated[di] = tr

        for ti, tr in enumerate(self.tracks):
            if ti not in matched_tracks:
                tr.missed += 1
        self.tracks = [tr for tr in self.tracks if tr.missed <= self.max_missed]

        for di, (_, center, radius) in enumerate(detections):
            if di not in matched_detections:
                tr = Track(center, radius, self._time)
                self.tracks.append(tr)
                associated[di] = tr

        # DWA still receives all LiDAR hits, including walls/uncertain shapes.
        for di, (indices, _, _) in enumerate(detections):
            tr = associated[di]
            if self._is_dynamic(tr):
                velocities[indices] = tr.vel
        return points, velocities

    def dynamic_tracks(self):
        return [tr for tr in self.tracks if self._is_dynamic(tr)]
