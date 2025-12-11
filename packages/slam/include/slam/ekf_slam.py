import numpy as np
import threading

class EkfSlam2D(object):
    def __init__(self):
        # State: [x, y, theta, l1_x, l1_y, l2_x, l2_y, ...]^T
        self.x = np.zeros((3, 1))
        self.P = np.eye(3) * 1e-3  # small initial uncertainty
        self.landmark_ids = []      # order of tag IDs in the state

        # Motion noise (for [x, y, theta])
        self.Q = np.diag([0.02, 0.02, np.deg2rad(2.0)]) ** 2

        # Measurement noise (for [range, bearing])
        self.R = np.diag([0.10, np.deg2rad(3.0)]) ** 2

        # Lock to prevent race conditions between predict and update
        self._lock = threading.Lock()

    def _get_landmark_index(self, tag_id):
        try:
            idx = self.landmark_ids.index(tag_id)
        except ValueError:
            idx = -1
        return idx

    def predict(self, v, w, dt):
        """
        v: linear velocity (m/s)
        w: angular velocity (rad/s)
        dt: time step (s)
        """
        if dt <= 0.0:
            return

        with self._lock:
            x = self.x
            n = x.shape[0]
            theta = x[2, 0]

            # Unicycle motion model
            if abs(w) < 1e-6:
                # straight line
                dx = v * dt * np.cos(theta)
                dy = v * dt * np.sin(theta)
                dtheta = 0.0
            else:
                dx = (v / w) * (np.sin(theta + w * dt) - np.sin(theta))
                dy = (v / w) * (-np.cos(theta + w * dt) + np.cos(theta))
                dtheta = w * dt

            # Update robot pose
            x[0, 0] += dx
            x[1, 0] += dy
            x[2, 0] = self._wrap_angle(theta + dtheta)

            # Jacobian of motion w.r.t robot state (3x3)
            G_rr = np.eye(3)
            if abs(w) < 1e-6:
                G_rr[0, 2] = -v * dt * np.sin(theta)
                G_rr[1, 2] = v * dt * np.cos(theta)
            else:
                G_rr[0, 2] = (v / w) * (np.cos(theta + w * dt) - np.cos(theta))
                G_rr[1, 2] = (v / w) * (np.sin(theta + w * dt) - np.sin(theta))

            # Embed into full-size G (landmarks don't move)
            G = np.eye(n)
            G[0:3, 0:3] = G_rr

            # Motion noise in full space (only affects robot pose)
            R_full = np.zeros((n, n))
            R_full[0:3, 0:3] = self.Q

            # Covariance propagation
            self.P = G @ self.P @ G.T + R_full

    def update(self, tag_id, z):
        """
        EKF update with a single AprilTag measurement.
        tag_id: integer ID of the observed tag
        z: np.array([range, bearing]) measured in robot frame
        """
        with self._lock:
            lm_index = self._get_landmark_index(tag_id)

            # New landmark? Initialize it.
            if lm_index == -1:
                self._initialize_landmark(tag_id, z)
                return  # Don't update on first observation - just initialize

            lm_start = 3 + 2 * lm_index
            lx = self.x[lm_start, 0]
            ly = self.x[lm_start + 1, 0]

            rx, ry, rtheta = self.x[0, 0], self.x[1, 0], self.x[2, 0]

            # Predicted measurement
            dx = lx - rx
            dy = ly - ry
            q = dx**2 + dy**2
            sqrt_q = np.sqrt(q)
            if sqrt_q < 1e-6:
                return  # avoid numerical issues

            z_hat = np.zeros((2, 1))
            z_hat[0, 0] = sqrt_q                      # range
            z_hat[1, 0] = self._wrap_angle(np.arctan2(dy, dx) - rtheta)  # bearing

            # Jacobian H (2 x state_dim)
            n = self.x.shape[0]
            H = np.zeros((2, n))

            # w.r.t robot pose
            H[0, 0] = -dx / sqrt_q
            H[0, 1] = -dy / sqrt_q
            H[0, 2] = 0.0

            H[1, 0] = dy / q
            H[1, 1] = -dx / q
            H[1, 2] = -1.0

            # w.r.t landmark position
            H[0, lm_start]     = dx / sqrt_q
            H[0, lm_start + 1] = dy / sqrt_q

            H[1, lm_start]     = -dy / q
            H[1, lm_start + 1] =  dx / q

            # Innovation
            y = z.reshape((2, 1)) - z_hat
            y[1, 0] = self._wrap_angle(y[1, 0])

            # Mahalanobis distance gate (reject outliers)
            S = H @ self.P @ H.T + self.R
            try:
                S_inv = np.linalg.inv(S)
            except np.linalg.LinAlgError:
                return  # Singular matrix, skip update
            
            mahal_dist_sq = float(y.T @ S_inv @ y)
            # Chi-squared threshold for 2 DOF at 99% confidence
            if mahal_dist_sq > 9.21:
                # Outlier - skip this measurement
                return

            K = self.P @ H.T @ S_inv

            # State update
            self.x = self.x + K @ y

            # Covariance update (Joseph form for numerical stability)
            I = np.eye(self.P.shape[0])
            IKH = I - K @ H
            self.P = IKH @ self.P @ IKH.T + K @ self.R @ K.T

            # Normalize yaw
            self.x[2, 0] = self._wrap_angle(self.x[2, 0])

    def _initialize_landmark(self, tag_id, z):
        """Add a new landmark to the state from a single [range, bearing] measurement."""
        r = float(z[0])
        b = float(z[1])
        rx, ry, rtheta = self.x[0, 0], self.x[1, 0], self.x[2, 0]

        # landmark position in world
        lx = rx + r * np.cos(rtheta + b)
        ly = ry + r * np.sin(rtheta + b)

        # Jacobian of landmark position w.r.t robot pose and measurement
        # lx = rx + r * cos(theta + b)
        # ly = ry + r * sin(theta + b)
        
        # Jacobian w.r.t [rx, ry, theta]: G_r
        G_r = np.zeros((2, 3))
        G_r[0, 0] = 1.0  # dlx/drx
        G_r[0, 1] = 0.0  # dlx/dry
        G_r[0, 2] = -r * np.sin(rtheta + b)  # dlx/dtheta
        G_r[1, 0] = 0.0  # dly/drx
        G_r[1, 1] = 1.0  # dly/dry
        G_r[1, 2] = r * np.cos(rtheta + b)   # dly/dtheta

        # Jacobian w.r.t [r, b]: G_z
        G_z = np.zeros((2, 2))
        G_z[0, 0] = np.cos(rtheta + b)  # dlx/dr
        G_z[0, 1] = -r * np.sin(rtheta + b)  # dlx/db
        G_z[1, 0] = np.sin(rtheta + b)  # dly/dr
        G_z[1, 1] = r * np.cos(rtheta + b)   # dly/db

        # Extend state vector
        self.x = np.vstack([self.x, np.array([[lx], [ly]])])

        # Extend covariance with proper cross-correlations
        n_old = self.P.shape[0]
        P_new = np.zeros((n_old + 2, n_old + 2))
        
        # Copy old covariance
        P_new[0:n_old, 0:n_old] = self.P

        # Robot pose covariance (3x3 block)
        P_rr = self.P[0:3, 0:3]

        # New landmark covariance: propagate uncertainty from robot pose and measurement
        # P_ll = G_r @ P_rr @ G_r.T + G_z @ R @ G_z.T
        P_ll = G_r @ P_rr @ G_r.T + G_z @ self.R @ G_z.T
        
        # Add some extra uncertainty for initialization robustness
        P_ll += np.eye(2) * 0.1**2
        
        P_new[n_old:, n_old:] = P_ll

        # Cross-covariance between robot and new landmark
        # P_rl = P_rr @ G_r.T (3x2)
        P_rl = self.P[0:3, 0:3] @ G_r.T
        P_new[0:3, n_old:] = P_rl
        P_new[n_old:, 0:3] = P_rl.T

        # Cross-covariance between existing landmarks and new landmark
        # For each existing landmark, the cross-covariance propagates through the robot
        for i in range(len(self.landmark_ids)):
            lm_start = 3 + 2 * i
            # P_li_new = P_li_r @ G_r.T
            P_li_r = self.P[lm_start:lm_start+2, 0:3]  # (2x3)
            P_li_new = P_li_r @ G_r.T  # (2x2)
            P_new[lm_start:lm_start+2, n_old:] = P_li_new
            P_new[n_old:, lm_start:lm_start+2] = P_li_new.T

        self.P = P_new
        self.landmark_ids.append(tag_id)

    @staticmethod
    def _wrap_angle(a):
        return (a + np.pi) % (2 * np.pi) - np.pi