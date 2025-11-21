import numpy as np

class EkfSlam2D(object):
    def __init__(self):
        # State: [x, y, theta, l1_x, l1_y, l2_x, l2_y, ...]^T
        self.x = np.zeros((3, 1))
        self.P = np.eye(3) * 1e-3  # small initial uncertainty
        self.landmark_ids = []      # order of tag IDs in the state

        # Motion noise (for [x, y, theta])
        self.Q = np.diag([0.01, 0.01, np.deg2rad(1.0)]) ** 2

        # Measurement noise (for [range, bearing])
        self.R = np.diag([0.05, np.deg2rad(2.0)]) ** 2

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

        x = self.x
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
        G_rr[0, 2] = -v * dt * np.sin(theta)
        G_rr[1, 2] = v * dt * np.cos(theta)

        # Embed into full-size G
        G = np.eye(self.x.shape[0])
        G[0:3, 0:3] = G_rr

        # Motion noise in full space
        R_full = np.zeros_like(self.P)
        R_full[0:3, 0:3] = self.Q

        # Covariance propagation
        self.P = G @ self.P @ G.T + R_full

    def update(self, tag_id, z):
        """
        EKF update with a single AprilTag measurement.
        tag_id: integer ID of the observed tag
        z: np.array([range, bearing]) measured in robot frame
        """
        lm_index = self._get_landmark_index(tag_id)

        # New landmark? Initialize it.
        if lm_index == -1:
            self._initialize_landmark(tag_id, z)
            lm_index = len(self.landmark_ids) - 1

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
        H = np.zeros((2, self.x.shape[0]))

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

        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        # State update
        self.x = self.x + K @ y

        # Covariance update
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ H) @ self.P

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

        # Extend state vector
        self.x = np.vstack([self.x, np.array([[lx], [ly]])])

        # Extend covariance
        n_old = self.P.shape[0]
        P_new = np.zeros((n_old + 2, n_old + 2))
        P_new[0:n_old, 0:n_old] = self.P

        # High initial uncertainty on landmark position
        P_new[n_old:, n_old:] = np.diag([0.5, 0.5]) ** 2

        self.P = P_new
        self.landmark_ids.append(tag_id)

    @staticmethod
    def _wrap_angle(a):
        return (a + np.pi) % (2 * np.pi) - np.pi
