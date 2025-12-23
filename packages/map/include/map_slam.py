import numpy as np
import threading

# TODO : review apriltag_map_slam_node.py
# TODO : add covariance calculations

class MapSlam2D(object):
    def __init__(self):
        self._lock = threading.Lock()
        self.x = np.zeros(3) # x: [x, y, theta] current pose
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
        }] # list of dictionaries containing pose and detection information
        self.landmark_ids = [] # list of tag_ids
        self.last_optimized_state = None
        self.last_optimized_landmark_ids = None

    def add_pose(self, v, w, dt, detections):
        """
        v: linear velocity (m/s)
        w: angular velocity (rad/s)
        dt: time step (s)
        detections: list of apriltag detections
        """
        if dt <= 0.0:
            return

        with self._lock:
            x = self.x
            theta = x[2]

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
            x[0] += dx
            x[1] += dy
            x[2] = self._wrap_angle(theta + dtheta)

            # pose information
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

            # landmark information
            for detection in detections:
                t = detection.pose_t
                cam_x = float(t[0, 0])  # right
                cam_y = float(t[1, 0])  # down
                cam_z = float(t[2, 0])  # forward

                # Convert to 2D robot frame: x forward, y left
                dx = cam_z
                dy = -cam_x

                r = np.sqrt(dx**2 + dy**2)
                bearing = np.arctan2(dy, dx)

                pose['observations'].append({
                    'landmark_id': detection.tag_id,
                    'range': r,
                    'bearing': bearing
                })

                tag_id = detection.tag_id
                if tag_id not in self.landmark_ids:
                    self.landmark_ids.append(tag_id)

            self.poses.append(pose)

            print(f"Added pose {len(self.poses)-1} with {len(detections)} detections.\n\n")

    def get_state(self):
        """Get state vector: [x_1, y_1, theta_1, x_2, y_2, theta_2, ..., l1_x, l1_y, l2_x, l2_y, ...]^T as a first heuristic"""
        # NOTE: only called from update(), which already holds the lock.
        num_poses = len(self.poses)
        num_landmarks = len(self.landmark_ids)
        state_dim = 3 * num_poses + 2 * num_landmarks

        # Initialize state vector
        state = np.zeros(3 * num_poses + 2 * num_landmarks)

        #Start off from last optimized state if available
        prev = None
        if self.last_optimized_state is not None:
            prev = np.asarray(self.last_optimized_state).ravel()
            # Copy as much as fits (prefix copy). This reuses old poses/landmarks that still exist.
            ncopy = min(prev.size, state_dim)
            if ncopy > 0:
                state[:ncopy] = prev[:ncopy]

            # If dimensions match exactly, we can just return a clean copy
            if prev.size == state_dim:
                # Wrap all pose angles and return
                for k in range(num_poses):
                    state[3*k + 2] = self._wrap_angle(state[3*k + 2])
                return state

        # Initial state
        state[0] = self.poses[0]['x']
        state[1] = self.poses[0]['y']
        state[2] = self.poses[0]['theta']

        # Next states
        for i in range(1, num_poses):
            odometry = self.poses[i]['odometry']
            
            # Update pose in state
            state[i * 3] = state[(i - 1) * 3] + odometry['dx']
            state[i * 3 + 1] = state[(i - 1) * 3 + 1] + odometry['dy']
            state[i * 3 + 2] = self._wrap_angle(state[(i - 1) * 3 + 2] + odometry['dtheta'])

        # Landmark states
        for i, pose in enumerate(self.poses):

            # pose
            x = state[i * 3]
            y = state[i * 3 + 1]
            theta = state[i * 3 + 2]

            for observation in pose['observations']:
                tag_id = observation['landmark_id']
                lm_index = self._get_landmark_index(tag_id)

                x_absolute = x + observation['range'] * np.cos(observation['bearing'] + theta)
                y_absolute = y + observation['range'] * np.sin(observation['bearing'] + theta)

                # Update state with latest landmark value
                state[num_poses * 3 + lm_index * 2] = x_absolute
                state[num_poses * 3 + lm_index * 2 + 1] = y_absolute

        return state

    def update(self, max_iter=50):
        """Maximum a posteriori optimization using Gauss-Newton update"""
        print("[MapSlam2D] Optimization started.")
        with self._lock:
            num_poses = len(self.poses)
            num_landmarks = len(self.landmark_ids)
            
            # get state vector (this may add new landmarks)
            state = self.get_state()
            print(f"Initial state vector : {state}\n\n")

            # Recompute num_landmarks after get_state, to match the actual state size
            num_landmarks = len(self.landmark_ids)

            # Prior for first pose
            x0_prior = state[0:3].copy()
            lambda_prior = 1e4

            # number of residuals to calculate
            n_residuals = (num_poses - 1) * 3 + sum(len(p['observations']) for p in self.poses) * 2

            for iteration in range(max_iter):

                residual_i = 0
                residuals = np.zeros((n_residuals))
                jacobian = np.zeros((n_residuals, num_poses * 3 + num_landmarks * 2))

                # Odometry
                for k in range(1, num_poses):
                    # Indices in state vector
                    i = (k - 1) * 3  # pose t - 1 in state
                    j = k * 3        # pose t in state

                    # Odometry stored with pose k (motion from k-1 -> k)
                    odometry = self.poses[k]['odometry']

                    # x prediction
                    x_i = state[i]
                    x_j = state[j]
                    x_pred = x_i + odometry['dx']

                    # y prediction
                    y_i = state[i + 1]
                    y_j = state[j + 1]
                    y_pred = y_i + odometry['dy']

                    # theta prediction
                    theta_i = state[i + 2]
                    theta_j = state[j + 2]
                    theta_pred = theta_i + odometry['dtheta']

                    # residuals
                    residuals[residual_i] = x_pred - x_j
                    residuals[residual_i + 1] = y_pred - y_j
                    residuals[residual_i + 2] = self._wrap_angle(theta_pred - theta_j)

                    # jacobian
                    jacobian[residual_i, i] = 1 # d(residual_x)/dx_i
                    jacobian[residual_i, j] = -1 # d(residual_x)/dx_j
                    jacobian[residual_i + 1, i + 1] = 1 # d(residual_y)/dy_i
                    jacobian[residual_i + 1, j + 1] = -1 # d(residual_y)/dy_j
                    jacobian[residual_i + 2, i + 2] = 1 # d(residual_theta)/dtheta_i
                    jacobian[residual_i + 2, j + 2] = -1 # d(residual_theta)/dtheta_j

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
                        tag_id = observation['landmark_id']
                        l_i = self._get_landmark_index(tag_id)
                        # Landmark indices in state vector
                        j = num_poses * 3 + l_i * 2
                        j_x = j
                        j_y = j + 1

                        l_x = state[j_x]
                        l_y = state[j_y]

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
                            residuals[residual_i] = r_pred - observation['range']
                            residuals[residual_i + 1] = self._wrap_angle(b_pred - observation['bearing'])

                            # jacobian (range) – correct signs and indices
                            jacobian[residual_i, p_i] = -dx / sqrt_q          # d(res_r)/d(p_x)
                            jacobian[residual_i, p_i + 1] = -dy / sqrt_q      # d(res_r)/d(p_y)
                            jacobian[residual_i, j_x] = dx / sqrt_q           # d(res_r)/d(l_x)
                            jacobian[residual_i, j_y] = dy / sqrt_q           # d(res_r)/d(l_y)

                            jacobian[residual_i + 1, p_i] = dy / q            # d(res_b)/d(p_x)
                            jacobian[residual_i + 1, p_i + 1] = -dx / q       # d(res_b)/d(p_y)
                            jacobian[residual_i + 1, p_i + 2] = -1            # d(res_b)/d(p_theta)
                            jacobian[residual_i + 1, j_x] = -dy / q           # d(res_b)/d(l_x)
                            jacobian[residual_i + 1, j_y] = dx / q            # d(res_b)/d(l_y)

                            residual_i += 2


                mu = 1e-6  # start small
                JTJ = jacobian.T @ jacobian
                JTR = jacobian.T @ residuals
                JTJ += mu * np.eye(JTJ.shape[0])

                #First pose:
                JTJ[0:3, 0:3] += lambda_prior * np.eye(3)
                JTR[0:3] += lambda_prior * (state[0:3] - x0_prior)


                # solver : JTJ * delta = JTR
                delta = np.linalg.solve(JTJ, -JTR)

                # update
                state += delta

                for k in range(num_poses):
                    state[3*k + 2] = self._wrap_angle(state[3*k + 2])

                # convergence check
                if np.linalg.norm(delta) < 1e-4:
                    print(f"[MapSlam2D] Converged at iteration {iteration}.")   
                    break
                
            self.last_optimized_state = state
            self.last_optimized_landmark_ids = list(self.landmark_ids)

        print("[MapSlam2D] Optimization finished.")
        return state

    def _get_landmark_index(self, tag_id):
        try:
            idx = self.landmark_ids.index(tag_id)
        except ValueError:
            idx = -1
        return idx

    @staticmethod
    def _wrap_angle(a):
        return (a + np.pi) % (2 * np.pi) - np.pi
