import numpy as np
import threading

class EkfSlam2D:
    """2D EKF-SLAM with AprilTag landmarks. State: [x, y, theta, l1_x, l1_y, ...]"""

    def __init__(self):
        self.x = np.zeros((3, 1))
        self.P = np.eye(3) * 1e-3
        self.landmark_ids = []

        # Noise parameters
        # self.Q = np.diag([0.08, 0.08, np.deg2rad(8.0)]) ** 2  # motion (for real life)
        self.Q = np.diag([0.02, 0.02, np.deg2rad(2.0)]) ** 2  # motion (for simulation)
        # self.R = np.diag([0.10, np.deg2rad(3.5)]) ** 2        # measurement (for real life)
        self.R = np.diag([0.05, np.deg2rad(3.0)]) ** 2        # measurement (for simulation)

        # Mahalanobis gate
        self.mahal_gate = 75
        
        # Initial landmark uncertainty
        # Not too high (causes huge ellipses) but not too low (overconfident)
        # 0.5m std dev = 0.25 variance is reasonable
        self.initial_landmark_variance = 0.25

        # Diagnostics
        self.last_update_tag_id = None
        self.last_update_accepted = None
        self.last_update_mahal_dist_sq = None
        self.last_update_was_new_landmark = None
        self.last_z = None
        self.last_z_hat = None
        self.last_innovation = None

        # Lock to prevent race conditions between predict and update
        self._lock = threading.Lock()

    def _get_landmark_index(self, tag_id):
        try:
            return self.landmark_ids.index(tag_id)
        except ValueError:
            return -1
        
    def _get_measurement_noise(self, range_m):
        """Dynamic measurement noise based on distance."""
        sigma_range = 0.05 + 0.10 * range_m
        sigma_bearing = np.deg2rad(3.0 + range_m)
        return np.diag([sigma_range, sigma_bearing]) ** 2

    def predict(self, v, w, dt):
        """
        Unicycle motion model prediction.
        v: linear velocity (m/s)
        w: angular velocity (rad/s)
        dt: time step (s)
        """
        if dt <= 0.0:
            return

        with self._lock:
            x = self.x
            n = x.shape[0]
            theta = float(x[2, 0])
            
            w_threshold = 1e-3

            # Unicycle motion model
            if abs(w) < w_threshold:
                dx = v * dt * np.cos(theta)
                dy = v * dt * np.sin(theta)
                dtheta = 0.0
                
                G_rr = np.array([
                    [1.0, 0.0, -v * dt * np.sin(theta)],
                    [0.0, 1.0,  v * dt * np.cos(theta)],
                    [0.0, 0.0, 1.0]
                ])
            else:
                # Arc motion
                ratio = v / w
                dx = ratio * (np.sin(theta + w * dt) - np.sin(theta))
                dy = ratio * (-np.cos(theta + w * dt) + np.cos(theta))
                dtheta = w * dt
                
                G_rr = np.array([
                    [1.0, 0.0, ratio * (np.cos(theta + w * dt) - np.cos(theta))],
                    [0.0, 1.0, ratio * (np.sin(theta + w * dt) - np.sin(theta))],
                    [0.0, 0.0, 1.0]
                ])

            # Update robot pose
            x[0, 0] += dx
            x[1, 0] += dy
            x[2, 0] = self._wrap_angle(theta + dtheta)

            self.x[0, 0] += dx
            self.x[1, 0] += dy
            self.x[2, 0] = self._wrap_angle(theta + dtheta)

            # Full Jacobian
            G = np.eye(n)
            G[0:3, 0:3] = G_rr

            # Process noise (only on robot state)
            Q_full = np.zeros((n, n))
            Q_full[0:3, 0:3] = self.Q * dt  # scale with dt

            self.P = G @ self.P @ G.T + Q_full

    def update(self, tag_id, z):
        """
        EKF update with a single AprilTag measurement.
        tag_id: integer ID of the observed tag
        z: np.array([range, bearing]) measured in robot frame
        """
        with self._lock:
            z_arr = np.array(z, dtype=float).reshape((2, 1))
            
            self.last_update_tag_id = int(tag_id)
            self.last_update_mahal_dist_sq = None
            self.last_update_accepted = None
            self.last_update_was_new_landmark = None
            self.last_z = z_arr.copy()
            self.last_z_hat = None
            self.last_innovation = None

            lm_index = self._get_landmark_index(tag_id)

            # New landmark? Initialize it.
            if lm_index == -1:
                self._initialize_landmark(tag_id, z)
                self.last_update_accepted = True
                self.last_update_was_new_landmark = True
                return

            accepted = self._do_update(lm_index, z_arr)
            self.last_update_accepted = bool(accepted)
            self.last_update_was_new_landmark = False

    def _do_update(self, lm_index, z):
        """Perform EKF update. Returns True if accepted."""
        lm_start = 3 + 2 * lm_index
        lx, ly = float(self.x[lm_start, 0]), float(self.x[lm_start + 1, 0])
        rx, ry, rtheta = float(self.x[0, 0]), float(self.x[1, 0]), float(self.x[2, 0])

        delta_x = lx - rx
        delta_y = ly - ry
        q = delta_x**2 + delta_y**2
        sqrt_q = np.sqrt(q)
        
        if sqrt_q < 1e-9:
            return False

        # Predicted measurement
        z_hat = np.array([
            [sqrt_q],
            [self._wrap_angle(np.arctan2(delta_y, delta_x) - rtheta)]
        ])
        self.last_z_hat = z_hat.copy()

         Measurement Jacobian
        n = self.x.shape[0]
        H = np.zeros((2, n))
        
        H[0, 0] = -delta_x / sqrt_q
        H[0, 1] = -delta_y / sqrt_q
        H[1, 0] =  delta_y / q
        H[1, 1] = -delta_x / q
        H[1, 2] = -1.0
        
        H[0, lm_start] =  delta_x / sqrt_q
        H[0, lm_start + 1] =  delta_y / sqrt_q
        H[1, lm_start] = -delta_y / q
        H[1, lm_start + 1] =  delta_x / q

        # Innovation
        y = z - z_hat
        y[1, 0] = self._wrap_angle(y[1, 0])
        self.last_innovation = y.copy()

        # Innovation covariance
        S = H @ self.P @ H.T + self.R

        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return False

        mahal_dist_sq = float(y.T @ S_inv @ y)
        self.last_update_mahal_dist_sq = mahal_dist_sq

        if self.use_mahal_gate and mahal_dist_sq > self.mahal_gate:
            return False

        # Kalman gain
        K = self.P @ H.T @ S_inv

        # State update
        self.x = self.x + K @ y

        # Covariance update (simple form)
        I = np.eye(n)
        self.P = (I - K @ H) @ self.P

        # Ensure symmetry
        self.P = (self.P + self.P.T) / 2

        self.x[2, 0] = self._wrap_angle(float(self.x[2, 0]))
        
        return True

    def _initialize_landmark(self, tag_id, z):
        """Initialize a new landmark."""
        r = float(z[0, 0])
        b = float(z[1, 0])
        rx, ry, rtheta = float(self.x[0, 0]), float(self.x[1, 0]), float(self.x[2, 0])

        # Landmark position in world frame
        lx = rx + r * np.cos(rtheta + b)
        ly = ry + r * np.sin(rtheta + b)

        # Expand state
        self.x = np.vstack([self.x, [[lx], [ly]]])

        # Expand covariance with proper initialization
        n_old = self.P.shape[0]
        P_new = np.zeros((n_old + 2, n_old + 2))
        P_new[:n_old, :n_old] = self.P

        # Jacobian of landmark position w.r.t robot pose
        G_r = np.array([
            [1.0, 0.0, -r * np.sin(rtheta + b)],
            [0.0, 1.0,  r * np.cos(rtheta + b)]
        ])

        # Jacobian of landmark position w.r.t measurement
        G_z = np.array([
            [np.cos(rtheta + b), -r * np.sin(rtheta + b)],
            [np.sin(rtheta + b),  r * np.cos(rtheta + b)]
        ])

        # Landmark covariance from robot uncertainty and measurement uncertainty
        P_ll = G_r @ self.P[:3, :3] @ G_r.T + G_z @ self.R @ G_z.T
        
        # Add small initial uncertainty (but not huge)
        P_ll += np.eye(2) * self.initial_landmark_variance
        
        P_new[n_old:, n_old:] = P_ll

        # Cross-covariance robot-landmark
        P_rl = self.P[:3, :3] @ G_r.T
        P_new[:3, n_old:] = P_rl
        P_new[n_old:, :3] = P_rl.T

        # Cross-covariance existing landmarks to new landmark
        for i in range(len(self.landmark_ids)):
            lm_start_i = 3 + 2 * i
            P_li_new = self.P[lm_start_i:lm_start_i + 2, :3] @ G_r.T
            P_new[lm_start_i:lm_start_i + 2, n_old:] = P_li_new
            P_new[n_old:, lm_start_i:lm_start_i + 2] = P_li_new.T

        self.P = P_new
        self.landmark_ids.append(int(tag_id))

    @staticmethod
    def _wrap_angle(a):
        return (a + np.pi) % (2.0 * np.pi) - np.pi