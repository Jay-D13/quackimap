"""
Graph-based MAP SLAM for 2D robot with AprilTag landmarks.

This implements Maximum A Posteriori estimation via Gauss-Newton optimization
on a factor graph containing odometry and landmark constraints.

Key fix: current_pose is now properly synchronized with optimized keyframes
to prevent path/robot divergence.
"""
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from typing import List, Dict, Optional
import rospy


class LightweightGraphSlam2D(object):
    
    def __init__(self, window_size=50, keyframe_distance=0.1, keyframe_angle=np.deg2rad(10),
                 compute_covariances=False):
        self.window_size = window_size
        self.keyframe_distance = keyframe_distance
        self.keyframe_angle = keyframe_angle
        self.compute_covariances = compute_covariances
        
        # State
        self.keyframe_poses: List[np.ndarray] = []
        self.landmarks: Dict[int, np.ndarray] = {}
        self.landmark_ids: List[int] = []
        
        # Covariances (only populated if compute_covariances=True)
        self.landmark_covariances: Dict[int, np.ndarray] = {}
        
        # Constraints
        self.odometry_constraints = []
        self.landmark_constraints = []
        
        # Noise parameters (information form uses inverse)
        self.Q = np.diag([0.02, 0.02, np.deg2rad(2.0)]) ** 2  # Odometry noise
        self.R = np.diag([0.10, np.deg2rad(3.0)]) ** 2        # Measurement noise
        
        # Optimization parameters
        self.max_iterations = 10
        self.convergence_threshold = 1e-4
        
        # Pose tracking
        self.current_pose = np.array([0.0, 0.0, 0.0], dtype=np.float64) # continuously updated with odometry 
        self.keyframe_poses.append(self.current_pose.copy())
        self.pose_at_last_keyframe = self.current_pose.copy() # snapshot when last keyframe was added
        
        self.delta_since_keyframe = np.array([0.0, 0.0, 0.0], dtype=np.float64) # accumulated motion since last keyframe
        
        # Performance tracking
        self.last_optimization_time = 0.0
        self.optimization_count = 0
        
    def _is_valid(self, arr: np.ndarray) -> bool:
        """Check for NaN/Inf."""
        return np.all(np.isfinite(arr))
    
    def should_add_keyframe(self) -> bool:
        """Check if current pose warrants a new keyframe."""
        if not self._is_valid(self.current_pose) or not self._is_valid(self.pose_at_last_keyframe):
            return False
            
        dx = self.current_pose[0] - self.pose_at_last_keyframe[0]
        dy = self.current_pose[1] - self.pose_at_last_keyframe[1]
        dist = np.sqrt(dx**2 + dy**2)
        dtheta = abs(self._wrap_angle(self.current_pose[2] - self.pose_at_last_keyframe[2]))
        
        return dist > self.keyframe_distance or dtheta > self.keyframe_angle
        
    def add_odometry_constraint(self, v: float, w: float, dt: float):
        """Add odometry measurement and update pose."""
        if dt <= 0.0 or dt > 1.0:
            return
        if not np.isfinite(v) or not np.isfinite(w):
            return
            
        theta = self.current_pose[2]
        
        # Unicycle motion model
        if abs(w) < 1e-6:
            dx = v * dt * np.cos(theta)
            dy = v * dt * np.sin(theta)
            dtheta = 0.0
        else:
            dx = (v / w) * (np.sin(theta + w * dt) - np.sin(theta))
            dy = (v / w) * (-np.cos(theta + w * dt) + np.cos(theta))
            dtheta = w * dt
        
        new_pose = np.array([
            self.current_pose[0] + dx,
            self.current_pose[1] + dy,
            self._wrap_angle(self.current_pose[2] + dtheta)
        ])
        
        if self._is_valid(new_pose):
            self.current_pose = new_pose
            # Track accumulated delta since last keyframe
            self.delta_since_keyframe[0] += dx
            self.delta_since_keyframe[1] += dy
            self.delta_since_keyframe[2] = self._wrap_angle(self.delta_since_keyframe[2] + dtheta)
        
        if self.should_add_keyframe():
            self._add_keyframe()
            
    def _add_keyframe(self):
        """Add current pose as keyframe with odometry constraint."""
        if not self._is_valid(self.current_pose):
            return
            
        prev_pose = self.pose_at_last_keyframe
        curr_pose = self.current_pose.copy()
        
        # Compute relative transformation in LOCAL frame of previous pose
        dx_world = curr_pose[0] - prev_pose[0]
        dy_world = curr_pose[1] - prev_pose[1]
        
        cos_t = np.cos(prev_pose[2])
        sin_t = np.sin(prev_pose[2])
        dx_local = cos_t * dx_world + sin_t * dy_world
        dy_local = -sin_t * dx_world + cos_t * dy_world
        dtheta = self._wrap_angle(curr_pose[2] - prev_pose[2])
        
        self.keyframe_poses.append(curr_pose)
        
        if len(self.keyframe_poses) >= 2:
            self.odometry_constraints.append({
                'from_idx': len(self.keyframe_poses) - 2,
                'to_idx': len(self.keyframe_poses) - 1,
                'delta': np.array([dx_local, dy_local, dtheta]),
            })
        
        # Reset tracking for next keyframe
        self.pose_at_last_keyframe = curr_pose.copy()
        self.delta_since_keyframe = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        
        if len(self.keyframe_poses) > self.window_size:
            self._marginalize_old_poses()
            
    def _marginalize_old_poses(self):
        """Remove old poses from sliding window."""
        n_to_remove = len(self.keyframe_poses) - self.window_size
        if n_to_remove <= 0:
            return
        
        self.keyframe_poses = self.keyframe_poses[n_to_remove:]
        
        # Update odometry constraint indices
        self.odometry_constraints = [
            {**c, 'from_idx': c['from_idx'] - n_to_remove, 'to_idx': c['to_idx'] - n_to_remove}
            for c in self.odometry_constraints
            if c['from_idx'] >= n_to_remove and c['to_idx'] >= n_to_remove
        ]
        
        # Update landmark constraint indices
        self.landmark_constraints = [
            {**c, 'pose_idx': c['pose_idx'] - n_to_remove}
            for c in self.landmark_constraints
            if c['pose_idx'] >= n_to_remove
        ]
        
    def add_landmark_observation(self, tag_id: int, range_meas: float, bearing_meas: float):
        """Add landmark observation."""
        if not self._is_valid(self.current_pose):
            return
        if not np.isfinite(range_meas) or not np.isfinite(bearing_meas):
            return
        if range_meas <= 0.01 or range_meas > 10.0:
            return
        
        keyframe_idx = len(self.keyframe_poses) - 1
        curr_pose = self.keyframe_poses[keyframe_idx]
        
        # Initialize new landmark
        if tag_id not in self.landmarks:
            lx = curr_pose[0] + range_meas * np.cos(curr_pose[2] + bearing_meas)
            ly = curr_pose[1] + range_meas * np.sin(curr_pose[2] + bearing_meas)
            
            if self._is_valid(np.array([lx, ly])):
                self.landmarks[tag_id] = np.array([lx, ly])
                self.landmark_ids.append(tag_id)
                self.landmark_covariances[tag_id] = np.diag([0.5, 0.5]) ** 2
                rospy.loginfo(f"New landmark {tag_id} at ({lx:.2f}, {ly:.2f})")
        
        self.landmark_constraints.append({
            'pose_idx': keyframe_idx,
            'tag_id': tag_id,
            'measurement': np.array([range_meas, bearing_meas]),
        })
    
    def optimize(self) -> bool:
        """Run Gauss-Newton optimization."""
        import time
        start_time = time.time()
        
        n_poses = len(self.keyframe_poses)
        n_landmarks = len(self.landmark_ids)
        
        if n_poses < 2:
            return False
        
        # Rate limiting
        if n_poses > 80:
            if time.time() - self.last_optimization_time < 2.0:
                return False
        
        pose_dim = 3 * n_poses
        lm_dim = 2 * n_landmarks
        state_dim = pose_dim + lm_dim
        
        Q_inv = np.linalg.inv(self.Q)
        R_inv = np.linalg.inv(self.R)
        
        # Backup for rollback
        backup_poses = [p.copy() for p in self.keyframe_poses]
        backup_landmarks = {k: v.copy() for k, v in self.landmarks.items()}
        
        # Store the last keyframe BEFORE optimization
        last_kf_before = self.keyframe_poses[-1].copy()
        dx_to_current = self.current_pose[0] - last_kf_before[0]
        dy_to_current = self.current_pose[1] - last_kf_before[1]
        dtheta_to_current = self._wrap_angle(self.current_pose[2] - last_kf_before[2])
        
        cos_t = np.cos(last_kf_before[2])
        sin_t = np.sin(last_kf_before[2])
        dx_local = cos_t * dx_to_current + sin_t * dy_to_current
        dy_local = -sin_t * dx_to_current + cos_t * dy_to_current
        
        for iteration in range(self.max_iterations):
            H = lil_matrix((state_dim, state_dim))
            b = np.zeros(state_dim)
            
            # Anchor first pose
            anchor_info = 1e8
            H[0, 0] = anchor_info
            H[1, 1] = anchor_info
            H[2, 2] = anchor_info
            
            # Odometry constraints
            for c in self.odometry_constraints:
                i, j = c['from_idx'], c['to_idx']
                z_ij = c['delta']
                
                pose_i = self.keyframe_poses[i]
                pose_j = self.keyframe_poses[j]
                
                dx_w = pose_j[0] - pose_i[0]
                dy_w = pose_j[1] - pose_i[1]
                
                cos_t = np.cos(pose_i[2])
                sin_t = np.sin(pose_i[2])
                
                # Predicted relative transformation in frame i
                h_ij = np.array([
                    cos_t * dx_w + sin_t * dy_w,
                    -sin_t * dx_w + cos_t * dy_w,
                    self._wrap_angle(pose_j[2] - pose_i[2])
                ])
                
                error = h_ij - z_ij
                error[2] = self._wrap_angle(error[2])
                
                # Jacobians
                J_i = np.array([
                    [-cos_t, -sin_t, -sin_t * dx_w + cos_t * dy_w],
                    [sin_t, -cos_t, -cos_t * dx_w - sin_t * dy_w],
                    [0, 0, -1]
                ])
                J_j = np.array([
                    [cos_t, sin_t, 0],
                    [-sin_t, cos_t, 0],
                    [0, 0, 1]
                ])
                
                idx_i, idx_j = 3 * i, 3 * j
                
                H[idx_i:idx_i+3, idx_i:idx_i+3] += J_i.T @ Q_inv @ J_i
                H[idx_i:idx_i+3, idx_j:idx_j+3] += J_i.T @ Q_inv @ J_j
                H[idx_j:idx_j+3, idx_i:idx_i+3] += J_j.T @ Q_inv @ J_i
                H[idx_j:idx_j+3, idx_j:idx_j+3] += J_j.T @ Q_inv @ J_j
                
                b[idx_i:idx_i+3] += J_i.T @ Q_inv @ error
                b[idx_j:idx_j+3] += J_j.T @ Q_inv @ error
            
            # Landmark constraints
            for c in self.landmark_constraints:
                pose_idx = c['pose_idx']
                tag_id = c['tag_id']
                z = c['measurement']
                
                if tag_id not in self.landmarks:
                    continue
                
                pose = self.keyframe_poses[pose_idx]
                lm_idx = self.landmark_ids.index(tag_id)
                lm = self.landmarks[tag_id]
                
                if not self._is_valid(pose) or not self._is_valid(lm):
                    continue
                
                dx = lm[0] - pose[0]
                dy = lm[1] - pose[1]
                q = dx**2 + dy**2
                
                if q < 1e-6:
                    continue
                    
                sqrt_q = np.sqrt(q)
                
                # Predicted measurement [range, bearing]
                h = np.array([sqrt_q, self._wrap_angle(np.arctan2(dy, dx) - pose[2])])
                error = z - h
                error[1] = self._wrap_angle(error[1])
                
                # Jacobian w.r.t. pose
                H_pose = np.array([
                    [-dx/sqrt_q, -dy/sqrt_q, 0],
                    [dy/q, -dx/q, -1]
                ])
                
                # Jacobian w.r.t. landmark
                H_lm = np.array([
                    [dx/sqrt_q, dy/sqrt_q],
                    [-dy/q, dx/q]
                ])
                
                p_idx = 3 * pose_idx
                l_idx = pose_dim + 2 * lm_idx
                
                H[p_idx:p_idx+3, p_idx:p_idx+3] += H_pose.T @ R_inv @ H_pose
                H[l_idx:l_idx+2, l_idx:l_idx+2] += H_lm.T @ R_inv @ H_lm
                H[p_idx:p_idx+3, l_idx:l_idx+2] += H_pose.T @ R_inv @ H_lm
                H[l_idx:l_idx+2, p_idx:p_idx+3] += H_lm.T @ R_inv @ H_pose
                
                b[p_idx:p_idx+3] += H_pose.T @ R_inv @ error
                b[l_idx:l_idx+2] += H_lm.T @ R_inv @ error
            
            # Regularization for numerical stability
            for i in range(state_dim):
                H[i, i] += 1e-6
            
            # Solve sparse system
            H_csr = H.tocsr()
            try:
                delta = spsolve(H_csr, -b)
            except Exception as e:
                rospy.logerr(f"Solver failed: {e}")
                self.keyframe_poses = backup_poses
                self.landmarks = backup_landmarks
                return False
            
            if not np.all(np.isfinite(delta)):
                rospy.logerr(f"NaN in solution at iteration {iteration}")
                self.keyframe_poses = backup_poses
                self.landmarks = backup_landmarks
                return False
            
            # Limit step size for stability
            delta = np.clip(delta, -0.5, 0.5)
            
            # Update poses
            for i in range(n_poses):
                idx = 3 * i
                self.keyframe_poses[i][0] += delta[idx]
                self.keyframe_poses[i][1] += delta[idx + 1]
                self.keyframe_poses[i][2] = self._wrap_angle(self.keyframe_poses[i][2] + delta[idx + 2])
            
            # Update landmarks
            for i, tag_id in enumerate(self.landmark_ids):
                idx = pose_dim + 2 * i
                self.landmarks[tag_id][0] += delta[idx]
                self.landmarks[tag_id][1] += delta[idx + 1]
            
            # Check convergence
            if np.linalg.norm(delta) < self.convergence_threshold:
                break
        
        last_kf_after = self.keyframe_poses[-1]
        
        # Transform the saved local delta back to world frame using NEW keyframe pose
        cos_t_new = np.cos(last_kf_after[2])
        sin_t_new = np.sin(last_kf_after[2])
        
        dx_world_new = cos_t_new * dx_local - sin_t_new * dy_local
        dy_world_new = sin_t_new * dx_local + cos_t_new * dy_local
        
        # Reconstruct current_pose: optimized_keyframe + rotated_delta
        self.current_pose = np.array([
            last_kf_after[0] + dx_world_new,
            last_kf_after[1] + dy_world_new,
            self._wrap_angle(last_kf_after[2] + dtheta_to_current)
        ])
        
        self.pose_at_last_keyframe = last_kf_after.copy()
        
        # Extract covariances if requested
        if self.compute_covariances and n_landmarks > 0:
            self._extract_landmark_covariances(H_csr, pose_dim, n_landmarks)
        
        elapsed = time.time() - start_time
        self.last_optimization_time = time.time()
        self.optimization_count += 1
        rospy.loginfo(f"MAP-SLAM Optimization: {iteration+1} iters, {elapsed:.3f}s, {n_poses} poses, {n_landmarks} landmarks")
        
        return True
    
    def _extract_landmark_covariances(self, H: csr_matrix, pose_dim: int, n_landmarks: int):
        """Extract marginal covariances for landmarks."""
        try:
            lm_start = pose_dim
            lm_end = pose_dim + 2 * n_landmarks
            
            H_lm = H[lm_start:lm_end, lm_start:lm_end].toarray()
            H_lm += np.eye(H_lm.shape[0]) * 1e-6
            
            Sigma_lm = np.linalg.inv(H_lm)
            
            for i, tag_id in enumerate(self.landmark_ids):
                idx = 2 * i
                self.landmark_covariances[tag_id] = Sigma_lm[idx:idx+2, idx:idx+2]
                
        except Exception as e:
            rospy.logwarn_throttle(10.0, f"Covariance extraction failed: {e}")
    
    def get_current_pose(self) -> np.ndarray:
        """Get current pose (synchronized with optimization)."""
        if self._is_valid(self.current_pose):
            return self.current_pose.copy()
        return np.array([0.0, 0.0, 0.0])
    
    def get_landmark_position(self, tag_id: int) -> Optional[np.ndarray]:
        if tag_id in self.landmarks and self._is_valid(self.landmarks[tag_id]):
            return self.landmarks[tag_id].copy()
        return None
    
    def get_landmark_covariance(self, tag_id: int) -> Optional[np.ndarray]:
        if tag_id in self.landmark_covariances:
            return self.landmark_covariances[tag_id].copy()
        return None
    
    def get_all_poses(self) -> List[np.ndarray]:
        """Get all optimized keyframe poses."""
        return [p.copy() for p in self.keyframe_poses if self._is_valid(p)]
    
    def get_all_landmarks(self) -> Dict[int, np.ndarray]:
        return {k: v.copy() for k, v in self.landmarks.items() if self._is_valid(v)}
    
    def get_all_landmark_covariances(self) -> Dict[int, np.ndarray]:
        return {k: v.copy() for k, v in self.landmark_covariances.items()}
    
    @staticmethod
    def _wrap_angle(angle: float) -> float:
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle