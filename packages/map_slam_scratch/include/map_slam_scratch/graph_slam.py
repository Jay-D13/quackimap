#!/usr/bin/env python3
"""
From-scratch Graph-based MAP SLAM using Gauss-Newton optimization.

This implements the same algorithm as GTSAM but without the library dependency,
making it suitable for educational purposes and embedded deployment.

The optimization solves:
    x* = argmin_x sum_i ||h_i(x) - z_i||^2_{Sigma_i}

where:
    - x is the state vector (poses + landmarks)
    - h_i(x) are the measurement prediction functions
    - z_i are the actual measurements
    - Sigma_i are the measurement covariances

We use Gauss-Newton with optional Levenberg-Marquardt damping.
"""

import math
import threading
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum

from .sparse_solver import SparseSolver


def wrap_angle(a: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class FactorType(Enum):
    PRIOR = 1
    ODOMETRY = 2
    LANDMARK = 3


@dataclass
class Factor:
    """Represents a factor (constraint) in the factor graph."""
    factor_type: FactorType
    var_indices: List[int]      # Indices into the state vector
    measurement: np.ndarray     # The measurement z
    info_matrix: np.ndarray     # Information matrix (inverse covariance)
    huber_k: float = 1.5        # Huber kernel parameter for robustness


@dataclass
class Pose2D:
    """2D pose representation."""
    x: float
    y: float
    theta: float
    
    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.theta])
    
    @staticmethod
    def from_array(arr: np.ndarray) -> 'Pose2D':
        return Pose2D(float(arr[0]), float(arr[1]), float(arr[2]))
    
    def compose(self, other: 'Pose2D') -> 'Pose2D':
        """Compose this pose with another (this * other)."""
        c = math.cos(self.theta)
        s = math.sin(self.theta)
        x = self.x + c * other.x - s * other.y
        y = self.y + s * other.x + c * other.y
        theta = wrap_angle(self.theta + other.theta)
        return Pose2D(x, y, theta)
    
    def inverse(self) -> 'Pose2D':
        """Compute the inverse of this pose."""
        c = math.cos(self.theta)
        s = math.sin(self.theta)
        x = -c * self.x - s * self.y
        y = s * self.x - c * self.y
        theta = -self.theta
        return Pose2D(x, y, wrap_angle(theta))
    
    def transform_point(self, px: float, py: float) -> Tuple[float, float]:
        """Transform a point from local to global frame."""
        c = math.cos(self.theta)
        s = math.sin(self.theta)
        gx = self.x + c * px - s * py
        gy = self.y + s * px + c * py
        return gx, gy


class GraphSlam2D:
    """
    Graph-based SLAM with Gauss-Newton/Levenberg-Marquardt optimization.
    
    State vector layout:
        [x0, y0, theta0, x1, y1, theta1, ..., lm0_x, lm0_y, lm1_x, lm1_y, ...]
        
    Each pose takes 3 elements, each landmark takes 2 elements.
    """
    
    def __init__(
        self,
        odom_sigmas: Tuple[float, float, float] = (0.15, 0.15, math.radians(12.0)),
        meas_sigmas: Tuple[float, float] = (0.20, math.radians(6.0)),
        prior_sigmas: Tuple[float, float, float] = (1e-3, 1e-3, 1e-3),
        huber_k: float = 1.5,
        enable_gating: bool = True,
        gate_chi2: float = 25.0,
        use_lm_damping: bool = True,
        lm_lambda_init: float = 1e-4,
    ):
        """
        Initialize the graph SLAM system.
        
        Args:
            odom_sigmas: (sigma_x, sigma_y, sigma_theta) for odometry
            meas_sigmas: (sigma_range, sigma_bearing) for landmark measurements
            prior_sigmas: Prior uncertainty on first pose
            huber_k: Huber kernel parameter for robust estimation
            enable_gating: Whether to reject outlier measurements
            gate_chi2: Chi-squared threshold for gating (2 DOF)
            use_lm_damping: Use Levenberg-Marquardt damping
            lm_lambda_init: Initial LM damping parameter
        """
        self.lock = threading.RLock()
        
        # Noise parameters
        self.odom_info = np.diag(1.0 / np.array(odom_sigmas)**2)
        self.meas_info = np.diag(1.0 / np.array(meas_sigmas)**2)
        self.prior_info = np.diag(1.0 / np.array(prior_sigmas)**2)
        
        self.huber_k = huber_k
        self.enable_gating = enable_gating
        self.gate_chi2 = gate_chi2
        
        # LM parameters
        self.use_lm_damping = use_lm_damping
        self.lm_lambda = lm_lambda_init
        self.lm_lambda_factor = 10.0
        
        # State
        self.num_poses = 0
        self.landmark_ids: Dict[int, int] = {}  # tag_id -> landmark_index
        self.state = np.zeros(0)  # Will grow as we add poses/landmarks
        
        # Factor graph
        self.factors: List[Factor] = []
        
        # Solver
        self.solver = SparseSolver(use_sparse=True, regularization=1e-8)
        
        # Initialize with prior on first pose at origin
        self._init_first_pose()
    
    def _state_dim(self) -> int:
        """Current dimension of state vector."""
        return 3 * self.num_poses + 2 * len(self.landmark_ids)
    
    def _pose_index(self, pose_id: int) -> int:
        """Get starting index of pose in state vector."""
        return 3 * pose_id
    
    def _landmark_index(self, lm_idx: int) -> int:
        """Get starting index of landmark in state vector."""
        return 3 * self.num_poses + 2 * lm_idx
    
    def _get_pose(self, pose_id: int) -> Pose2D:
        """Extract pose from state vector."""
        idx = self._pose_index(pose_id)
        return Pose2D.from_array(self.state[idx:idx+3])
    
    def _set_pose(self, pose_id: int, pose: Pose2D):
        """Set pose in state vector."""
        idx = self._pose_index(pose_id)
        self.state[idx:idx+3] = pose.to_array()
    
    def _get_landmark(self, lm_idx: int) -> Tuple[float, float]:
        """Extract landmark position from state vector."""
        idx = self._landmark_index(lm_idx)
        return float(self.state[idx]), float(self.state[idx+1])
    
    def _set_landmark(self, lm_idx: int, x: float, y: float):
        """Set landmark position in state vector."""
        idx = self._landmark_index(lm_idx)
        self.state[idx] = x
        self.state[idx+1] = y
    
    def _init_first_pose(self):
        """Initialize the first pose at origin with a prior factor."""
        with self.lock:
            # Add first pose
            self.state = np.zeros(3)
            self.num_poses = 1
            
            # Add prior factor
            prior = Factor(
                factor_type=FactorType.PRIOR,
                var_indices=[0],  # First pose
                measurement=np.array([0.0, 0.0, 0.0]),
                info_matrix=self.prior_info.copy(),
                huber_k=self.huber_k,
            )
            self.factors.append(prior)
    
    def last_pose(self) -> Pose2D:
        """Get the current (last) pose estimate."""
        with self.lock:
            return self._get_pose(self.num_poses - 1)
    
    def get_all_poses(self) -> List[Tuple[float, float, float]]:
        """Get all pose estimates."""
        with self.lock:
            poses = []
            for i in range(self.num_poses):
                p = self._get_pose(i)
                poses.append((p.x, p.y, p.theta))
            return poses
    
    def get_all_landmarks(self) -> List[Tuple[int, float, float]]:
        """Get all landmark estimates with their tag IDs."""
        with self.lock:
            landmarks = []
            for tag_id, lm_idx in self.landmark_ids.items():
                x, y = self._get_landmark(lm_idx)
                landmarks.append((tag_id, x, y))
            return landmarks
    
    def add_odometry(self, dx: float, dy: float, dtheta: float):
        """
        Add an odometry constraint between the last pose and a new pose.
        
        Args:
            dx, dy: Translation in the local frame of the previous pose
            dtheta: Rotation (radians)
        """
        with self.lock:
            prev_pose_id = self.num_poses - 1
            new_pose_id = self.num_poses
            
            # Expand state vector
            old_state = self.state.copy()
            self.state = np.zeros(3 * (new_pose_id + 1) + 2 * len(self.landmark_ids))
            self.state[:len(old_state)] = old_state
            
            # Initialize new pose by composing
            prev_pose = self._get_pose(prev_pose_id)
            delta = Pose2D(dx, dy, dtheta)
            new_pose = prev_pose.compose(delta)
            self._set_pose(new_pose_id, new_pose)
            
            # Move landmarks to new positions in state vector
            for tag_id, lm_idx in self.landmark_ids.items():
                old_idx = 3 * self.num_poses + 2 * lm_idx
                new_idx = 3 * (self.num_poses + 1) + 2 * lm_idx
                if old_idx < len(old_state):
                    self.state[new_idx:new_idx+2] = old_state[old_idx:old_idx+2]
            
            self.num_poses += 1
            
            # Add odometry factor
            factor = Factor(
                factor_type=FactorType.ODOMETRY,
                var_indices=[prev_pose_id, new_pose_id],
                measurement=np.array([dx, dy, dtheta]),
                info_matrix=self.odom_info.copy(),
                huber_k=self.huber_k,
            )
            self.factors.append(factor)
            
            # Optimize incrementally
            self._optimize(max_iters=3)
    
    def add_tag_measurement(self, tag_id: int, rng: float, bearing: float) -> Tuple[bool, float]:
        """
        Add a landmark measurement.
        
        Args:
            tag_id: Unique identifier for the landmark
            rng: Measured range to landmark
            bearing: Measured bearing to landmark (radians)
            
        Returns:
            (accepted, mahal_dist_sq): Whether measurement was accepted and its Mahalanobis distance
        """
        with self.lock:
            current_pose = self._get_pose(self.num_poses - 1)
            
            # Check if landmark is new
            is_new = tag_id not in self.landmark_ids
            
            if is_new:
                # Initialize new landmark
                lm_idx = len(self.landmark_ids)
                self.landmark_ids[tag_id] = lm_idx
                
                # Expand state vector
                old_state = self.state.copy()
                self.state = np.zeros(3 * self.num_poses + 2 * (lm_idx + 1))
                self.state[:len(old_state)] = old_state
                
                # Initialize landmark position from measurement
                lm_x = current_pose.x + rng * math.cos(current_pose.theta + bearing)
                lm_y = current_pose.y + rng * math.sin(current_pose.theta + bearing)
                self._set_landmark(lm_idx, lm_x, lm_y)
            else:
                lm_idx = self.landmark_ids[tag_id]
                
                # Gating: check if measurement is consistent with current estimate
                if self.enable_gating:
                    lm_x, lm_y = self._get_landmark(lm_idx)
                    dx = lm_x - current_pose.x
                    dy = lm_y - current_pose.y
                    pred_rng = math.hypot(dx, dy)
                    pred_bearing = wrap_angle(math.atan2(dy, dx) - current_pose.theta)
                    
                    # Compute Mahalanobis distance
                    err = np.array([rng - pred_rng, wrap_angle(bearing - pred_bearing)])
                    mahal_sq = err @ self.meas_info @ err
                    
                    if mahal_sq > self.gate_chi2:
                        return False, mahal_sq
            
            # Add landmark factor
            factor = Factor(
                factor_type=FactorType.LANDMARK,
                var_indices=[self.num_poses - 1, tag_id],  # pose_id and tag_id
                measurement=np.array([rng, bearing]),
                info_matrix=self.meas_info.copy(),
                huber_k=self.huber_k,
            )
            self.factors.append(factor)
            
            # Optimize incrementally
            self._optimize(max_iters=3)
            
            return True, 0.0
    
    def _compute_prior_residual(self, factor: Factor) -> Tuple[np.ndarray, np.ndarray]:
        """Compute residual and Jacobian for prior factor."""
        pose_id = factor.var_indices[0]
        pose = self._get_pose(pose_id)
        
        # Residual: h(x) - z = current_pose - prior
        residual = pose.to_array() - factor.measurement
        residual[2] = wrap_angle(residual[2])
        
        # Jacobian w.r.t pose: identity
        J = np.eye(3)
        
        return residual, J
    
    def _compute_odometry_residual(self, factor: Factor) -> Tuple[np.ndarray, np.ndarray]:
        """Compute residual and Jacobian for odometry factor."""
        pose_i_id = factor.var_indices[0]
        pose_j_id = factor.var_indices[1]
        
        pose_i = self._get_pose(pose_i_id)
        pose_j = self._get_pose(pose_j_id)
        delta_meas = Pose2D.from_array(factor.measurement)
        
        # Predicted delta: pose_i^(-1) * pose_j
        c = math.cos(pose_i.theta)
        s = math.sin(pose_i.theta)
        
        dx = pose_j.x - pose_i.x
        dy = pose_j.y - pose_i.y
        
        pred_dx = c * dx + s * dy
        pred_dy = -s * dx + c * dy
        pred_dtheta = wrap_angle(pose_j.theta - pose_i.theta)
        
        # Residual
        residual = np.array([
            pred_dx - delta_meas.x,
            pred_dy - delta_meas.y,
            wrap_angle(pred_dtheta - delta_meas.theta)
        ])
        
        # Jacobians
        # J_i (w.r.t pose_i) and J_j (w.r.t pose_j)
        J = np.zeros((3, 6))
        
        # d(residual)/d(pose_i)
        J[0, 0] = -c
        J[0, 1] = -s
        J[0, 2] = -s * dx + c * dy
        J[1, 0] = s
        J[1, 1] = -c
        J[1, 2] = -c * dx - s * dy
        J[2, 2] = -1.0
        
        # d(residual)/d(pose_j)
        J[0, 3] = c
        J[0, 4] = s
        J[1, 3] = -s
        J[1, 4] = c
        J[2, 5] = 1.0
        
        return residual, J
    
    def _compute_landmark_residual(self, factor: Factor) -> Tuple[np.ndarray, np.ndarray]:
        """Compute residual and Jacobian for landmark measurement factor."""
        pose_id = factor.var_indices[0]
        tag_id = factor.var_indices[1]
        lm_idx = self.landmark_ids[tag_id]
        
        pose = self._get_pose(pose_id)
        lm_x, lm_y = self._get_landmark(lm_idx)
        
        # Predicted measurement
        dx = lm_x - pose.x
        dy = lm_y - pose.y
        q = dx**2 + dy**2
        sqrt_q = math.sqrt(q) if q > 1e-9 else 1e-9
        
        pred_range = sqrt_q
        pred_bearing = wrap_angle(math.atan2(dy, dx) - pose.theta)
        
        # Residual
        residual = np.array([
            pred_range - factor.measurement[0],
            wrap_angle(pred_bearing - factor.measurement[1])
        ])
        
        # Jacobian: [d(r)/d(pose), d(r)/d(landmark)]
        J = np.zeros((2, 5))  # 2 residuals, 3 pose + 2 landmark
        
        # d(residual)/d(pose)
        J[0, 0] = -dx / sqrt_q
        J[0, 1] = -dy / sqrt_q
        J[0, 2] = 0.0
        
        J[1, 0] = dy / q
        J[1, 1] = -dx / q
        J[1, 2] = -1.0
        
        # d(residual)/d(landmark)
        J[0, 3] = dx / sqrt_q
        J[0, 4] = dy / sqrt_q
        
        J[1, 3] = -dy / q
        J[1, 4] = dx / q
        
        return residual, J
    
    def _apply_huber(self, residual: np.ndarray, info: np.ndarray, k: float) -> Tuple[np.ndarray, np.ndarray]:
        """Apply Huber robust kernel to down-weight outliers."""
        # Compute Mahalanobis norm
        mahal = math.sqrt(residual @ info @ residual)
        
        if mahal <= k:
            # Quadratic region - no change
            return residual, info
        else:
            # Linear region - reduce weight
            weight = k / mahal
            return residual * math.sqrt(weight), info * weight
    
    def _build_linear_system(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Build the linear system H @ dx = -b from all factors.
        
        Returns:
            H: Information matrix
            b: Gradient vector
            total_error: Sum of squared errors
        """
        n = self._state_dim()
        H = np.zeros((n, n))
        b = np.zeros(n)
        total_error = 0.0
        
        for factor in self.factors:
            if factor.factor_type == FactorType.PRIOR:
                residual, J_local = self._compute_prior_residual(factor)
                pose_idx = self._pose_index(factor.var_indices[0])
                
                # Apply robust kernel
                r_robust, info_robust = self._apply_huber(residual, factor.info_matrix, factor.huber_k)
                
                # Build contributions
                JtS = J_local.T @ info_robust
                H[pose_idx:pose_idx+3, pose_idx:pose_idx+3] += JtS @ J_local
                b[pose_idx:pose_idx+3] += JtS @ r_robust
                
                total_error += r_robust @ info_robust @ r_robust
                
            elif factor.factor_type == FactorType.ODOMETRY:
                residual, J_local = self._compute_odometry_residual(factor)
                
                pose_i_idx = self._pose_index(factor.var_indices[0])
                pose_j_idx = self._pose_index(factor.var_indices[1])
                
                # Apply robust kernel
                r_robust, info_robust = self._apply_huber(residual, factor.info_matrix, factor.huber_k)
                
                # J_local is [3x6]: first 3 cols for pose_i, last 3 for pose_j
                J_i = J_local[:, :3]
                J_j = J_local[:, 3:]
                
                JtS_i = J_i.T @ info_robust
                JtS_j = J_j.T @ info_robust
                
                # H contributions
                H[pose_i_idx:pose_i_idx+3, pose_i_idx:pose_i_idx+3] += JtS_i @ J_i
                H[pose_i_idx:pose_i_idx+3, pose_j_idx:pose_j_idx+3] += JtS_i @ J_j
                H[pose_j_idx:pose_j_idx+3, pose_i_idx:pose_i_idx+3] += JtS_j @ J_i
                H[pose_j_idx:pose_j_idx+3, pose_j_idx:pose_j_idx+3] += JtS_j @ J_j
                
                # b contributions
                b[pose_i_idx:pose_i_idx+3] += JtS_i @ r_robust
                b[pose_j_idx:pose_j_idx+3] += JtS_j @ r_robust
                
                total_error += r_robust @ info_robust @ r_robust
                
            elif factor.factor_type == FactorType.LANDMARK:
                residual, J_local = self._compute_landmark_residual(factor)
                
                pose_id = factor.var_indices[0]
                tag_id = factor.var_indices[1]
                lm_idx = self.landmark_ids[tag_id]
                
                pose_idx = self._pose_index(pose_id)
                lm_global_idx = self._landmark_index(lm_idx)
                
                # Apply robust kernel
                r_robust, info_robust = self._apply_huber(residual, factor.info_matrix, factor.huber_k)
                
                # J_local is [2x5]: first 3 cols for pose, last 2 for landmark
                J_p = J_local[:, :3]
                J_l = J_local[:, 3:]
                
                JtS_p = J_p.T @ info_robust
                JtS_l = J_l.T @ info_robust
                
                # H contributions
                H[pose_idx:pose_idx+3, pose_idx:pose_idx+3] += JtS_p @ J_p
                H[pose_idx:pose_idx+3, lm_global_idx:lm_global_idx+2] += JtS_p @ J_l
                H[lm_global_idx:lm_global_idx+2, pose_idx:pose_idx+3] += JtS_l @ J_p
                H[lm_global_idx:lm_global_idx+2, lm_global_idx:lm_global_idx+2] += JtS_l @ J_l
                
                # b contributions
                b[pose_idx:pose_idx+3] += JtS_p @ r_robust
                b[lm_global_idx:lm_global_idx+2] += JtS_l @ r_robust
                
                total_error += r_robust @ info_robust @ r_robust
        
        return H, b, total_error
    
    def _optimize(self, max_iters: int = 10, tol: float = 1e-5):
        """
        Run Gauss-Newton (or Levenberg-Marquardt) optimization.
        
        Args:
            max_iters: Maximum number of iterations
            tol: Convergence tolerance on dx norm
        """
        for iteration in range(max_iters):
            H, b, error = self._build_linear_system()
            
            # Levenberg-Marquardt damping
            if self.use_lm_damping:
                H_damped = H + self.lm_lambda * np.diag(np.diag(H) + 1e-6)
            else:
                H_damped = H
            
            # Solve for update
            try:
                dx = self.solver.solve(H_damped, b)
            except Exception:
                break
            
            # Check convergence
            dx_norm = np.linalg.norm(dx)
            if dx_norm < tol:
                break
            
            # Apply update
            new_state = self.state + dx
            
            # Wrap angles for poses
            for i in range(self.num_poses):
                idx = self._pose_index(i) + 2
                new_state[idx] = wrap_angle(new_state[idx])
            
            # LM: check if error decreased
            if self.use_lm_damping:
                old_state = self.state.copy()
                self.state = new_state
                _, _, new_error = self._build_linear_system()
                
                if new_error < error:
                    # Accept update, decrease damping
                    self.lm_lambda /= self.lm_lambda_factor
                else:
                    # Reject update, increase damping
                    self.state = old_state
                    self.lm_lambda *= self.lm_lambda_factor
            else:
                self.state = new_state
    
    def batch_optimize(self, max_iters: int = 50) -> np.ndarray:
        """
        Run full batch optimization.
        
        Args:
            max_iters: Maximum number of iterations
            
        Returns:
            Optimized state vector
        """
        with self.lock:
            self._optimize(max_iters=max_iters, tol=1e-6)
            return self.state.copy()
