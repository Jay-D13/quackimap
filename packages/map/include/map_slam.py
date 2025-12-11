import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

class MapSlam2D(object):
    """
    Fixed version of the uploaded MAP SLAM implementation.
    
    Key fixes:
    1. Corrected landmark indexing bug
    2. Added information matrices (Q, R)
    3. Changed to sparse matrix operations
    4. Fixed Jacobian indices
    5. Separated state management from optimization
    """
    
    def __init__(self):
        self.x = np.zeros(3)  # Current pose: [x, y, theta]
        self.poses = [{
            'x': self.x[0],
            'y': self.x[1],
            'theta': self.x[2],
            'odometry': {
                'dx': 0.0,
                'dy': 0.0,
                'dtheta': 0.0
            },
            'observations': []
        }]
        self.landmark_ids = []
        
        # Store optimized state for queries
        self.optimized_state = None
        
        # Information matrices (inverse of covariance)
        # Motion model uncertainty
        self.Q_odom = np.linalg.inv(np.diag([0.01, 0.01, np.deg2rad(1.0)]) ** 2)
        
        # Measurement uncertainty  
        self.R_landmark = np.linalg.inv(np.diag([0.05, np.deg2rad(2.0)]) ** 2)

    def add_pose(self, v, w, dt, detections):
        """
        Add a new pose with odometry and observations.
        
        Args:
            v: linear velocity (m/s)
            w: angular velocity (rad/s)
            dt: time step (s)
            detections: list of apriltag detections
        """
        if dt <= 0.0:
            return

        x = self.x
        theta = x[2]

        # Unicycle motion model
        if abs(w) < 1e-6:
            dx = v * dt * np.cos(theta)
            dy = v * dt * np.sin(theta)
            dtheta = 0.0
        else:
            dx = (v / w) * (np.sin(theta + w * dt) - np.sin(theta))
            dy = (v / w) * (-np.cos(theta + w * dt) + np.cos(theta))
            dtheta = w * dt

        # Update current pose estimate
        x[0] += dx
        x[1] += dy
        x[2] = self._wrap_angle(theta + dtheta)

        # Create pose record
        pose = {
            'x': x[0],
            'y': x[1],
            'theta': x[2],
            'odometry': {
                'dx': dx,
                'dy': dy,
                'dtheta': dtheta
            },
            'observations': []
        }

        # Add landmark observations
        for detection in detections:
            t = detection.pose_t
            cam_x = float(t[0, 0])
            cam_y = float(t[1, 0])
            cam_z = float(t[2, 0])

            # Convert to robot frame
            dx_cam = cam_z
            dy_cam = -cam_x

            r = np.sqrt(dx_cam**2 + dy_cam**2)
            bearing = np.arctan2(dy_cam, dx_cam)

            pose['observations'].append({
                'landmark_id': detection.tag_id,
                'range': r,
                'bearing': bearing
            })
            
            # Initialize landmark if first observation
            if detection.tag_id not in self.landmark_ids:
                self.landmark_ids.append(detection.tag_id)

        self.poses.append(pose)

    def get_state(self):
        """
        Initialize state vector from poses and landmarks.
        Returns: [x_0, y_0, theta_0, ..., x_n, y_n, theta_n, l1_x, l1_y, ..., lm_x, lm_y]
        """
        num_poses = len(self.poses)
        num_landmarks = len(self.landmark_ids)

        state = np.zeros(3 * num_poses + 2 * num_landmarks)

        # Initialize poses from odometry
        state[0] = self.poses[0]['x']
        state[1] = self.poses[0]['y']
        state[2] = self.poses[0]['theta']

        for i in range(1, num_poses):
            odometry = self.poses[i]['odometry']
            state[i * 3] = state[(i - 1) * 3] + odometry['dx']
            state[i * 3 + 1] = state[(i - 1) * 3 + 1] + odometry['dy']
            state[i * 3 + 2] = self._wrap_angle(state[(i - 1) * 3 + 2] + odometry['dtheta'])

        # Initialize landmarks from first observation of each
        landmark_initialized = [False] * num_landmarks
        
        for i, pose in enumerate(self.poses):
            pose_x = state[i * 3]
            pose_y = state[i * 3 + 1]
            pose_theta = state[i * 3 + 2]
            
            for obs in pose['observations']:
                tag_id = obs['landmark_id']
                lm_idx = self.landmark_ids.index(tag_id)
                
                # Initialize landmark only once (from first observation)
                if not landmark_initialized[lm_idx]:
                    lm_state_idx = num_poses * 3 + lm_idx * 2
                    state[lm_state_idx] = pose_x + obs['range'] * np.cos(obs['bearing'] + pose_theta)
                    state[lm_state_idx + 1] = pose_y + obs['range'] * np.sin(obs['bearing'] + pose_theta)
                    landmark_initialized[lm_idx] = True

        return state

    def optimize(self, max_iter=20):
        """
        Perform MAP optimization using Gauss-Newton with sparse matrices.
        
        Returns:
            Optimized state vector
        """
        num_poses = len(self.poses)
        num_landmarks = len(self.landmark_ids)
        
        if num_poses < 2:
            return self.get_state()
        
        # Initialize state
        state = self.get_state()
        state_dim = len(state)

        for iteration in range(max_iter):
            # Count residuals
            n_residuals = (num_poses - 1) * 3  # Odometry constraints
            for pose in self.poses:
                n_residuals += len(pose['observations']) * 2  # Landmark constraints

            # Build sparse system
            H = lil_matrix((state_dim, state_dim))
            b = np.zeros(state_dim)
            
            residual_idx = 0

            # ===== ODOMETRY CONSTRAINTS =====
            for k in range(1, num_poses):
                i = (k - 1) * 3  # Previous pose index
                j = k * 3        # Current pose index

                odometry = self.poses[k]['odometry']

                # Predicted pose from odometry
                x_pred = state[i] + odometry['dx']
                y_pred = state[i + 1] + odometry['dy']
                theta_pred = self._wrap_angle(state[i + 2] + odometry['dtheta'])

                # Residuals (error = predicted - actual)
                e = np.array([
                    x_pred - state[j],
                    y_pred - state[j + 1],
                    self._wrap_angle(theta_pred - state[j + 2])
                ])

                # Jacobian for odometry constraint
                # J = [I_3x3, -I_3x3] where I is identity
                # This is because: e = (x_{i-1} + odom) - x_i
                
                # Information matrix weighted residual and Jacobian
                info = self.Q_odom
                
                # Contribution to H and b for pose i (if not first pose)
                if i > 0:  # Don't optimize first pose (it's fixed)
                    # H[i:i+3, i:i+3] += I^T @ info @ I = info
                    H[i:i+3, i:i+3] += info
                    # H[i:i+3, j:j+3] += I^T @ info @ (-I) = -info
                    H[i:i+3, j:j+3] += -info
                    # b[i:i+3] += I^T @ info @ e
                    b[i:i+3] += info @ e
                
                # Contribution for pose j
                # H[j:j+3, i:i+3] += (-I)^T @ info @ I = -info
                if i > 0:
                    H[j:j+3, i:i+3] += -info
                # H[j:j+3, j:j+3] += (-I)^T @ info @ (-I) = info
                H[j:j+3, j:j+3] += info
                # b[j:j+3] += (-I)^T @ info @ e = -info @ e
                b[j:j+3] += -info @ e

            # ===== LANDMARK CONSTRAINTS =====
            for pose_idx, pose in enumerate(self.poses):
                p_i = pose_idx * 3
                p_x = state[p_i]
                p_y = state[p_i + 1]
                p_theta = state[p_i + 2]

                for obs in pose['observations']:
                    tag_id = obs['landmark_id']
                    lm_idx = self.landmark_ids.index(tag_id)
                    lm_state_idx = num_poses * 3 + lm_idx * 2

                    l_x = state[lm_state_idx]
                    l_y = state[lm_state_idx + 1]

                    # Predicted measurement
                    dx = l_x - p_x
                    dy = l_y - p_y
                    q = dx**2 + dy**2
                    sqrt_q = np.sqrt(q)

                    if sqrt_q < 1e-6:
                        continue

                    r_pred = sqrt_q
                    b_pred = self._wrap_angle(np.arctan2(dy, dx) - p_theta)

                    # Residual
                    e = np.array([
                        r_pred - obs['range'],
                        self._wrap_angle(b_pred - obs['bearing'])
                    ])

                    # Jacobian w.r.t. robot pose (2x3)
                    H_pose = np.zeros((2, 3))
                    H_pose[0, 0] = -dx / sqrt_q  # dr/dx
                    H_pose[0, 1] = -dy / sqrt_q  # dr/dy
                    H_pose[0, 2] = 0.0           # dr/dtheta
                    H_pose[1, 0] = dy / q        # dbearing/dx
                    H_pose[1, 1] = -dx / q       # dbearing/dy
                    H_pose[1, 2] = -1.0          # dbearing/dtheta

                    # Jacobian w.r.t. landmark (2x2)
                    H_lm = np.zeros((2, 2))
                    H_lm[0, 0] = dx / sqrt_q     # dr/dl_x
                    H_lm[0, 1] = dy / sqrt_q     # dr/dl_y
                    H_lm[1, 0] = -dy / q         # dbearing/dl_x
                    H_lm[1, 1] = dx / q          # dbearing/dl_y

                    info = self.R_landmark

                    # Add to system (only if pose is not first)
                    if pose_idx > 0:
                        H[p_i:p_i+3, p_i:p_i+3] += H_pose.T @ info @ H_pose
                        H[p_i:p_i+3, lm_state_idx:lm_state_idx+2] += H_pose.T @ info @ H_lm
                        b[p_i:p_i+3] += H_pose.T @ info @ e

                    # Landmark contributions
                    H[lm_state_idx:lm_state_idx+2, lm_state_idx:lm_state_idx+2] += H_lm.T @ info @ H_lm
                    if pose_idx > 0:
                        H[lm_state_idx:lm_state_idx+2, p_i:p_i+3] += H_lm.T @ info @ H_pose
                    b[lm_state_idx:lm_state_idx+2] += H_lm.T @ info @ e

            # Solve sparse system
            H_csr = H.tocsr()
            try:
                delta = spsolve(H_csr, -b)
            except:
                print(f"Optimization failed at iteration {iteration}")
                return state

            # Update state
            state += delta
            
            # Wrap angles
            for i in range(num_poses):
                state[i * 3 + 2] = self._wrap_angle(state[i * 3 + 2])

            # Check convergence
            if np.linalg.norm(delta) < 1e-4:
                print(f"Converged in {iteration + 1} iterations")
                break

        # Store optimized state for landmark queries
        self.optimized_state = state
        
        # Update current pose estimate from optimized state
        if num_poses > 0:
            curr_idx = (num_poses - 1) * 3
            self.x[0] = state[curr_idx]
            self.x[1] = state[curr_idx + 1]
            self.x[2] = state[curr_idx + 2]

        return state

    def get_landmark_position(self, tag_id):
        """
        Get optimized landmark position.
        
        Args:
            tag_id: AprilTag ID
            
        Returns:
            numpy array [x, y] or None if landmark not found or not optimized yet
        """
        if self.optimized_state is None:
            return None
            
        try:
            lm_idx = self.landmark_ids.index(tag_id)
        except ValueError:
            return None
        
        num_poses = len(self.poses)
        lm_state_idx = num_poses * 3 + lm_idx * 2
        
        if lm_state_idx + 1 >= len(self.optimized_state):
            return None
        
        return np.array([
            self.optimized_state[lm_state_idx],
            self.optimized_state[lm_state_idx + 1]
        ])

    def _get_landmark_index(self, tag_id):
        """Get index of landmark in landmark_ids list."""
        try:
            idx = self.landmark_ids.index(tag_id)
        except ValueError:
            idx = -1
        return idx

    @staticmethod
    def _wrap_angle(a):
        """Wrap angle to [-pi, pi]."""
        return (a + np.pi) % (2 * np.pi) - np.pi