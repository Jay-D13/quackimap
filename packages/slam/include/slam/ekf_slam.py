import numpy as np
import threading

class EkfSlam2D(object):
    def __init__(self):
        # State: [x, y, theta, l1_x, l1_y, l2_x, l2_y, ...]^T
        self.x = np.zeros((3, 1))
        self.P = np.eye(3) * 1e-3  # small initial uncertainty
        self.landmark_ids = []      # order of tag IDs in the state

        # Motion noise (for [x, y, theta])
        self.Q = np.diag([0.08, 0.08, np.deg2rad(8.0)]) ** 2

        # Measurement noise (for [range, bearing])
        self.R = np.diag([0.10, np.deg2rad(3.0)]) ** 2

        # Mahalanobis gate threshold (chi-squared, 2 DOF). 99% ~= 9.21
        self.mahal_gate = 25 #9.21
        
        # if a measurement is rejected, inflate robot covariance and retry once
        # if > 1.0: gets enabled
        self.recovery_inflate_factor = 1.0

        # Diagnostics for the last update
        self.last_update_tag_id = None
        self.last_update_accepted = None
        self.last_update_mahal_dist_sq = None
        self.last_update_was_new_landmark = None

        # Lock to prevent race conditions between predict and update
        self._lock = threading.Lock()

    def _get_landmark_index(self, tag_id):
        try:
            return self.landmark_ids.index(tag_id)
        except ValueError:
            return -1

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
            theta = float(x[2, 0])

            # Unicycle motion model
            if abs(w) < 1e-6:
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
                G_rr[1, 2] =  v * dt * np.cos(theta)
            else:
                G_rr[0, 2] = (v / w) * (np.cos(theta + w * dt) - np.cos(theta))
                G_rr[1, 2] = (v / w) * (np.sin(theta + w * dt) - np.sin(theta))

            # Embed into full-size G (landmarks don't move)
            G = np.eye(n)
            G[0:3, 0:3] = G_rr

            # Discretize continuous-time process noise
            Qd = self.Q * float(dt)

            R_full = np.zeros((n, n))
            R_full[0:3, 0:3] = Qd

            self.P = G @ self.P @ G.T + R_full

    def update(self, tag_id, z):
        """
        EKF update with a single AprilTag measurement.
        tag_id: integer ID of the observed tag
        z: np.array([range, bearing]) measured in robot frame
        """
        with self._lock:
            self.last_update_tag_id = int(tag_id)
            self.last_update_mahal_dist_sq = None
            self.last_update_accepted = None
            self.last_update_was_new_landmark = None
            self.last_z = np.array(z, dtype=float).reshape((2, 1))
            self.last_z_hat = None
            self.last_innovation = None

            lm_index = self._get_landmark_index(tag_id)

            # New landmark? Initialize it.
            if lm_index == -1:
                self._initialize_landmark(tag_id, z)
                self.last_update_accepted = True
                self.last_update_was_new_landmark = True
                return

            accepted = self._do_update(lm_index, self.last_z)

            # Optional recovery: inflate pose covariance and retry once
            if (not accepted) and float(self.recovery_inflate_factor) > 1.0:
                f = float(self.recovery_inflate_factor)
                self.P[0:3, 0:3] *= f
                accepted = self._do_update(lm_index, self.last_z)

            self.last_update_accepted = bool(accepted)
            self.last_update_was_new_landmark = False

    def _do_update(self, lm_index, z):
        """Return True if update accepted."""
        lm_start = 3 + 2 * lm_index
        lx = float(self.x[lm_start, 0])
        ly = float(self.x[lm_start + 1, 0])

        rx, ry, rtheta = float(self.x[0, 0]), float(self.x[1, 0]), float(self.x[2, 0])

        dx = lx - rx
        dy = ly - ry
        q = dx**2 + dy**2
        sqrt_q = np.sqrt(q)
        if sqrt_q < 1e-9:
            return False

        z_hat = np.zeros((2, 1))
        z_hat[0, 0] = sqrt_q
        z_hat[1, 0] = self._wrap_angle(np.arctan2(dy, dx) - rtheta)

        self.last_z_hat = z_hat.copy()

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
        H[0, lm_start] = dx / sqrt_q
        H[0, lm_start + 1] = dy / sqrt_q
        H[1, lm_start] = -dy / q
        H[1, lm_start + 1] = dx / q

        y = z - z_hat
        y[1, 0] = self._wrap_angle(y[1, 0])
        self.last_innovation = y.copy()

        S = H @ self.P @ H.T + self.R
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return False

        mahal_dist_sq = float(y.T @ S_inv @ y)
        self.last_update_mahal_dist_sq = mahal_dist_sq

        if mahal_dist_sq > float(self.mahal_gate):
            return False

        K = self.P @ H.T @ S_inv

        self.x = self.x + K @ y

        I = np.eye(self.P.shape[0])
        IKH = I - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ self.R @ K.T

        self.x[2, 0] = self._wrap_angle(float(self.x[2, 0]))
        return True

    def _initialize_landmark(self, tag_id, z):
        r = float(z[0])
        b = float(z[1])
        rx, ry, rtheta = float(self.x[0, 0]), float(self.x[1, 0]), float(self.x[2, 0])

        lx = rx + r * np.cos(rtheta + b)
        ly = ry + r * np.sin(rtheta + b)

        # Jacobian w.r.t [rx, ry, theta]
        G_r = np.zeros((2, 3))
        G_r[0, 0] = 1.0
        G_r[0, 2] = -r * np.sin(rtheta + b)
        G_r[1, 1] = 1.0
        G_r[1, 2] =  r * np.cos(rtheta + b)

        # Jacobian w.r.t [r, b]
        G_z = np.zeros((2, 2))
        G_z[0, 0] = np.cos(rtheta + b)
        G_z[0, 1] = -r * np.sin(rtheta + b)
        G_z[1, 0] = np.sin(rtheta + b)
        G_z[1, 1] =  r * np.cos(rtheta + b)

        self.x = np.vstack([self.x, np.array([[lx], [ly]])])

        n_old = self.P.shape[0]
        P_new = np.zeros((n_old + 2, n_old + 2))
        P_new[0:n_old, 0:n_old] = self.P

        P_rr = self.P[0:3, 0:3]
        P_ll = G_r @ P_rr @ G_r.T + G_z @ self.R @ G_z.T

        # Extra uncertainty on first observation (helps prevent early over-confidence)
        P_ll += np.eye(2) * (0.15 ** 2)

        P_new[n_old:, n_old:] = P_ll

        # Cross-covariance robot <-> landmark
        P_rl = self.P[0:3, 0:3] @ G_r.T
        P_new[0:3, n_old:] = P_rl
        P_new[n_old:, 0:3] = P_rl.T

        # Cross-cov existing landmarks <-> new landmark through robot
        for i in range(len(self.landmark_ids)):
            lm_start = 3 + 2 * i
            P_li_r = self.P[lm_start:lm_start + 2, 0:3]
            P_li_new = P_li_r @ G_r.T
            P_new[lm_start:lm_start + 2, n_old:] = P_li_new
            P_new[n_old:, lm_start:lm_start + 2] = P_li_new.T

        self.P = P_new
        self.landmark_ids.append(int(tag_id))

    @staticmethod
    def _wrap_angle(a):
        return (a + np.pi) % (2.0 * np.pi) - np.pi