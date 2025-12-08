import numpy as np

class MapSlam2D(object):
    def __init__(self):
        # x: [x, y, theta]
        self.x = np.zeros((3, 1))
        self.poses = []

    def get_poses(self, v, w, dt, tag_id, z):
                """
        v: linear velocity (m/s)
        w: angular velocity (rad/s)
        dt: time step (s)
        tag_id: id of landmark
        z: [range, bearing] of landmark
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

        # pose and landmark information
        pose = {
            'x': x[0, 0],
            'y': x[1, 0],
            'theta': x[2, 0],
            'observations': {
                'landmark_id': tag_id,
                'range': float(z[0]),
                'bearing': float(z[1])
            }
        }

        self.poses.append(pose)

    def get_state(self, num_landmarks):
        """Get state vector: [x_1, y_1, theta_1, x_2, y_2, theta_2, ..., l1_x, l1_y, l2_x, l2_y, ...]^T"""
        # Initialize state
        state = np.zeros(len(self.poses) * 3 + num_landmarks * 2)

        # Initial state
        state[0] = self.poses[0]['x']
        state[1] = self.poses[0]['y']
        state[2] = self.poses[0]['theta']

        # Next states
        for i in range(1, len(self.poses)):
            odometry = self.poses[i - 1]['odometry']
            if odometry:
                state[i * 3] = state[(i - 1) * 3] + odometry['dx']
                state[i * 3 + 1] = state[(i - 1) * 3 + 1] + odometry['dy']
                state[i * 3 + 2] = state[(i - 1) * 3 + 2] + odometry['dtheta']

        # Landmark states
        for i, pose in enumerate(self.poses):
            for observation in pose['observations']:
                l_i = observation['landmark_id']
                l_x = state[i * 3] * np.cos(observation['bearing'] + state[i * 3 + 2])
                l_y = state[i * 3 + 1] * np.sin(observation['bearing'] + state[i * 3 + 2])
        # TODO : state vector landmarks

    def update(self, poses, num_landmarks, max_iter=50):
        """Maximum a posteriori optimization using Gauss-Newton update"""
        state = self.get_state(num_landmarks)

        for iteration in range(max_iter):

            residual_i = 0
            residuals = np.array([])
            jacobian = np.array((qqc, len(self.poses) * 3 + num_landmarks * 2))

            # Odometry
            for k in range(len(self.poses) - 1):
                i = k * 3        # pose i
                j = (k + 1) * 3  # pose i + 1

                # x prediction
                x_i = state[i]
                x_j = state[j]
                x_pred = x_j - x_i

                # y prediction
                y_i = state[i + 1]
                y_j = state[j + 1]
                y_pred = y_j - y_i

                # theta prediction
                theta_i = state[i + 2]
                theta_j = state[j + 2]
                theta_pred = theta_j - theta_i

                # residuals
                odometry = self.poses[i]['odometry']
                if odometry:
                    np.hstack((residuals, np.array([x_pred - odometry['dx']])))
                    np.hstack((residuals, np.array([y_pred - odometry['dy']])))
                    np.hstack((residuals, np.array([theta_pred - odometry['theta']])))

                # jacobian
                jacobian[residual_i, i] = -1
                jacobian[residual_i, j] = 1
                jacobian[residual_i + 1, i + 1] = -1
                jacobian[residual_i + 1, j + 1] = 1
                jacobian[residual_i + 2, i + 2] = -1
                jacobian[residual_i + 2, j + 2] = 1

                residual_i += 3

            # Landmarks
            for i, pose in enumerate(self.poses):
                # pose
                p_i = i * 3
                p_x = state[p_i]
                p_y = state[p_i + 1]
                p_theta = state[p_i + 2]

                for observation in pose['observations']:
                    # landmark
                    l_i = observation['landmark_id']
                    j = len(self.poses) * 3 + l_i * 2
                    l_x = state[j]
                    l_y = state[j + 1]

                    # landmark prediction
                    dx = l_x - p_x
                    dy = l_y - p_y
                    q = dx**2 + dy**2
                    sqrt_q = np.sqrt(q)

                    if sqrt_q > 1e-6:
                        # landmark prediction
                        r_pred = sqrt_q
                        b_pred = self._wrap_angle(np.arctan2(dy, dx) - p_theta)

                        # residuals
                        np.hstack((residuals, np.array([observation['range'] - r_pred])))
                        np.hstack((residuals, np.array([observation['bearing'] - b_pred])))

                        # jacobian (range)
                        jacobian[residual_i, p_i] = dx / sqrt_q
                        jacobian[residual_i, p_i + 1] = dy / sqrt_q
                        jacobian[residual_i, l_i] = -dx / sqrt_q
                        jacobian[residual_i, l_i + 1] = -dy / sqrt_q

                        # jacobian (bearing)
                        jacobian[residual_i + 1, p_i] = -dy / q
                        jacobian[residual_i + 1, p_i + 1] = dx / q
                        jacobian[residual_i + 1, p_i + 2] = 1
                        jacobian[residual_i + 1, l_i] = dy / q
                        jacobian[residual_i + 1, l_i + 1] = -dx / q

                        residual_i += 2
            
            JTJ = jacobian.T @ jacobian
            JTR = jacobian.T @ residuals

            H = JTJ

            # first pose
            H[0, 0] += 1000
            H[1, 1] += 1000
            H[2, 2] += 1000
            JTR[0:3] = 0

            # solver
            delta = np.linalg.solve(H, JTR)

            # update
            state += delta

            # convergence check
            if np.linalg.norm(delta) < 1e-6:
                break

        return state

    @staticmethod
    def _wrap_angle(a):
        return (a + np.pi) % (2 * np.pi) - np.pi