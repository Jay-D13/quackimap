#!/usr/bin/env python3
import math
import threading
import numpy as np
import gtsam


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def point_xy(p):
    """Return (x,y) from a GTSAM Point2
    depends on the GTSAM Python build
    """
    # Object-style
    if hasattr(p, "x") and callable(getattr(p, "x")):
        return float(p.x()), float(p.y())
    # Array-style (common)
    arr = np.asarray(p).reshape(-1)
    if arr.size >= 2:
        return float(arr[0]), float(arr[1])
    raise TypeError(f"Unsupported Point2 type: {type(p)}")


class MapSlam2D:
    """
    MAP-SLAM with:
      - Prior on first pose (anchors map frame)
      - BetweenFactorPose2 for odometry
      - BearingRangeFactor2D for tag measurements (data assoc by tag_id)
      - Robust noise (Huber) + optional simple chi2 gating
      - iSAM2 incremental + batch optimize on request
    """

    def __init__(
        self,
        odom_sigmas=(0.15, 0.15, np.deg2rad(12.0)),     # meters, meters, rad
        meas_sigmas=(0.20, np.deg2rad(6.0)),            # range(m), bearing(rad)
        prior_sigmas=(1e-3, 1e-3, 1e-3),
        huber_k=1.5,
        enable_gating=True,
        gate_chi2=25.0,
    ):
        self.lock = threading.RLock()

        self.odom_sigmas = np.array(odom_sigmas, dtype=float)
        self.meas_sigmas = np.array(meas_sigmas, dtype=float)  # [range, bearing]
        self.prior_sigmas = np.array(prior_sigmas, dtype=float)
        self.huber_k = float(huber_k)

        self.enable_gating = bool(enable_gating)
        self.gate_chi2 = float(gate_chi2)

        # Keys
        self.pose_idx = 0
        self.landmark_key_by_id = {}  # tag_id -> gtsam.Key

        # Full graph (for batch optimize) TODO haven't tested it tbh
        self.full_graph = gtsam.NonlinearFactorGraph()

        # iSAM2 incremental
        params = gtsam.ISAM2Params()
        params.setRelinearizeThreshold(0.1)
        params.relinearizeSkip = 1
        self.isam = gtsam.ISAM2(params)

        # Current estimate Values
        self.values = gtsam.Values()

        # Create prior on first pose at origin
        self._init_first_pose()

    def _robust_diag(self, sigmas, huber_k=None):
        base = gtsam.noiseModel.Diagonal.Sigmas(np.array(sigmas, dtype=float))
        k = self.huber_k if huber_k is None else float(huber_k)
        return gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Huber(k), base)

    def _init_first_pose(self):
        with self.lock:
            k0 = gtsam.symbol('x', 0)
            prior = gtsam.PriorFactorPose2(
                k0,
                gtsam.Pose2(0.0, 0.0, 0.0),
                gtsam.noiseModel.Diagonal.Sigmas(self.prior_sigmas),
            )
            self.full_graph.add(prior)
            self.values.insert(k0, gtsam.Pose2(0.0, 0.0, 0.0))

            graph_new = gtsam.NonlinearFactorGraph()
            init_new = gtsam.Values()
            graph_new.add(prior)
            init_new.insert(k0, gtsam.Pose2(0.0, 0.0, 0.0))
            self.isam.update(graph_new, init_new)
            self.values = self.isam.calculateEstimate()

    def _last_existing_pose_key(self):
        """Return the newest x<i> key that exists in self.values."""
        i = int(self.pose_idx)
        while i >= 0:
            k = gtsam.symbol('x', i)
            if self.values.exists(k):
                return k, i
            i -= 1
        return gtsam.symbol('x', 0), 0

    def last_pose(self) -> gtsam.Pose2:
        with self.lock:
            k, _ = self._last_existing_pose_key()
            return self.values.atPose2(k)

    def get_all_poses(self):
        with self.lock:
            poses = []
            _, last_i = self._last_existing_pose_key()
            for i in range(last_i + 1):
                k = gtsam.symbol('x', i)
                if self.values.exists(k):
                    p = self.values.atPose2(k)
                    poses.append((p.x(), p.y(), p.theta()))
            return poses

    def get_all_landmarks(self):
        with self.lock:
            out = []
            for tag_id, lkey in self.landmark_key_by_id.items():
                if not self.values.exists(lkey):
                    continue
                pt = self.values.atPoint2(lkey)
                lx, ly = point_xy(pt)
                out.append((int(tag_id), lx, ly))
            return out

    def add_odometry(self, dx: float, dy: float, dtheta: float):
        """Adds a new pose with BetweenFactor from previous pose."""
        with self.lock:
            prev_k = gtsam.symbol('x', int(self.pose_idx))
            next_idx = int(self.pose_idx) + 1
            curr_k = gtsam.symbol('x', next_idx)

            odom_meas = gtsam.Pose2(float(dx), float(dy), float(dtheta))
            noise = self._robust_diag(self.odom_sigmas)
            factor = gtsam.BetweenFactorPose2(prev_k, curr_k, odom_meas, noise)

            if not self.values.exists(prev_k):
                prev_k, _ = self._last_existing_pose_key()
            prev_pose = self.values.atPose2(prev_k)
            curr_init = prev_pose.compose(odom_meas)

            self.full_graph.add(factor)
            if not self.values.exists(curr_k):
                self.values.insert(curr_k, curr_init)

            graph_new = gtsam.NonlinearFactorGraph()
            init_new = gtsam.Values()
            graph_new.add(factor)
            init_new.insert(curr_k, curr_init)

            self.isam.update(graph_new, init_new)
            self.values = self.isam.calculateEstimate()

            self.pose_idx = next_idx

    def _ensure_landmark(self, tag_id: int, pose: gtsam.Pose2, rng: float, bearing: float):
        if tag_id in self.landmark_key_by_id:
            return self.landmark_key_by_id[tag_id], False

        lkey = gtsam.symbol('l', int(tag_id))
        lx_r = rng * math.cos(bearing)
        ly_r = rng * math.sin(bearing)
        init_pt = pose.transformFrom(gtsam.Point2(lx_r, ly_r))

        self.landmark_key_by_id[tag_id] = lkey
        if not self.values.exists(lkey):
            self.values.insert(lkey, init_pt)
        return lkey, True

    def _simple_gate(self, rng: float, bearing: float, pose: gtsam.Pose2, lpt) -> float:
        lx, ly = point_xy(lpt)
        dx = lx - pose.x()
        dy = ly - pose.y()
        pred_rng = math.hypot(dx, dy)
        pred_b = wrap_angle(math.atan2(dy, dx) - pose.theta())

        dr = (rng - pred_rng)
        db = wrap_angle(bearing - pred_b)

        sig_r = float(self.meas_sigmas[0])
        sig_b = float(self.meas_sigmas[1])
        return (dr / sig_r) ** 2 + (db / sig_b) ** 2

    def add_tag_measurement(self, tag_id: int, rng: float, bearing: float):
        with self.lock:
            xk, _ = self._last_existing_pose_key()
            pose = self.values.atPose2(xk)

            lkey, is_new = self._ensure_landmark(int(tag_id), pose, float(rng), float(bearing))
            lpt = self.values.atPoint2(lkey)

            if self.enable_gating and (not is_new):
                m2 = self._simple_gate(float(rng), float(bearing), pose, lpt)
                if m2 > self.gate_chi2:
                    return False, m2

            noise = self._robust_diag([self.meas_sigmas[1], self.meas_sigmas[0]])
            factor = gtsam.BearingRangeFactor2D(
                xk,
                lkey,
                gtsam.Rot2(float(bearing)),
                float(rng),
                noise,
            )

            self.full_graph.add(factor)

            graph_new = gtsam.NonlinearFactorGraph()
            init_new = gtsam.Values()
            graph_new.add(factor)
            if is_new:
                init_new.insert(lkey, self.values.atPoint2(lkey))

            self.isam.update(graph_new, init_new)
            self.values = self.isam.calculateEstimate()
            return True, 0.0

    def _reset_isam_from_current(self):
        self.isam = gtsam.ISAM2(self._isam_params)
        self.isam.update(self.full_graph, self.values)
        self.values = self.isam.calculateEstimate()

    def batch_optimize(self, max_iters=50):
        with self.lock:
            params = gtsam.LevenbergMarquardtParams()
            params.setMaxIterations(int(max_iters))
            opt = gtsam.LevenbergMarquardtOptimizer(self.full_graph, self.values, params)
            self.values = opt.optimize()
            self._reset_isam_from_current()
            return self.values