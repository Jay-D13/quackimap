#!/usr/bin/env python3
import rospy
import numpy as np
import cv2
import tf2_ros

import matplotlib
matplotlib.use('Agg')

from geometry_msgs.msg import TransformStamped, Point
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, CameraInfo, Image
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from visualization_msgs.msg import Marker, MarkerArray

from dt_apriltags import Detector
from ekf_slam.include.slam.ekf_slam import EkfSlam2D
from ekf_slam.include.slam.visualization import SlamVisualizer


class AprilTagEkfSlamNode(object):
    def __init__(self):
        # Parameters (can be overridden with rosparam)
        self.image_topic = rospy.get_param("~image_topic", "/camera/compressed")
        self.odom_topic = rospy.get_param("~odom_topic", "/odom")
        self.camera_info_topic = rospy.get_param("~camera_info_topic", "/camera_node/camera_info")
        self.tag_size = rospy.get_param("~tag_size", 0.065)
        self.map_name = rospy.get_param("~map_name", "loop")
        self.veh_name = rospy.get_param("~veh_name", "vehicle_0")
        self.wait_for_gt = rospy.get_param("~wait_for_gt", True)

        self.camera_params = None
        self.slam = EkfSlam2D()
        self.last_odom_time = None
        self.last_v, self.last_w = 0.0, 0.0
        self.detection_count = 0
        self.odom_count = 0

        # Path history
        self.path = Path()
        self.path.header.frame_id = "map"
        self.gt_path = Path()
        self.gt_path.header.frame_id = "map"

        # AprilTag detector
        self.detector = Detector(
            families='tag36h11', nthreads=1, quad_decimate=1,
            quad_sigma=0.0, refine_edges=1, decode_sharpening=0.25, debug=0
        )

        # Initialization
        self.slam_initialized = not self.wait_for_gt
        if self.wait_for_gt:
            rospy.loginfo("SLAM: Waiting for GT pose to initialize...")
            rospy.Timer(rospy.Duration(5.0), self._check_init_timeout, oneshot=True)

        # Visualizer
        self.visualizer = SlamVisualizer(self.map_name, self.veh_name)
        self.bridge = CvBridge()
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        # Subscribers
        self.camera_info_sub = rospy.Subscriber(self.camera_info_topic, CameraInfo, self._camera_info_cb, queue_size=1)
        rospy.Subscriber(self.image_topic, CompressedImage, self._image_cb, queue_size=1, buff_size=2**24)
        rospy.Subscriber(self.odom_topic, Odometry, self._odom_cb, queue_size=50)
        rospy.Subscriber(f"/{self.veh_name}/gt_pose", Odometry, self._gt_pose_cb, queue_size=10)

        # Publishers
        self.pose_pub = rospy.Publisher("slam_pose", PoseStamped, queue_size=10)
        self.pose_cov_pub = rospy.Publisher("slam_pose_cov", PoseWithCovarianceStamped, queue_size=10)
        self.lm_pub = rospy.Publisher("slam_landmarks", MarkerArray, queue_size=10)
        self.path_pub = rospy.Publisher("slam_path", Path, queue_size=10)
        self.slam_odom_pub = rospy.Publisher("slam_odom", Odometry, queue_size=10)
        self.gt_path_pub = rospy.Publisher("gt_path", Path, queue_size=10)
        self.debug_img_pub = rospy.Publisher("slam_debug_image/compressed", CompressedImage, queue_size=1)
        self.debug_img_raw_pub = rospy.Publisher("slam_debug_image", Image, queue_size=1)
        self.viz_pub = rospy.Publisher("slam_viz/compressed", CompressedImage, queue_size=1)

        # Visualization timer
        rospy.Timer(rospy.Duration(1.0), self._viz_timer_cb)

        rospy.loginfo("AprilTag EKF-SLAM node initialized")
        rospy.loginfo(f"  Image topic: {self.image_topic}")
        rospy.loginfo(f"  Odom topic: {self.odom_topic}")
        rospy.loginfo(f"  Tag size: {self.tag_size}m")
        rospy.loginfo(f"  Map: {self.map_name}, Vehicle: {self.veh_name}")
        rospy.loginfo(f"  GT landmarks loaded: {len(self.visualizer.gt_landmarks)}")

    def check_init_timeout(self, event):
        if not self.slam_initialized:
            rospy.logwarn("SLAM: Still waiting for GT pose... Is the topic correct?")
            rospy.logwarn(f"SLAM: Listening on /{self.veh_name}/gt_pose")
            rospy.logwarn("SLAM: You can disable this wait by setting ~wait_for_gt to False.")

    def camera_info_cb(self, msg: CameraInfo):
        # K is row-major: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
        K = msg.K
        fx = K[0]
        fy = K[4]
        cx = K[2]
        cy = K[5]
        self.camera_params = [fx, fy, cx, cy]

        rospy.loginfo(f"Got camera intrinsics fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
        self.camera_info_sub.unregister()

    # ---------- GROUND TRUTH CALLBACK ----------
    def gt_pose_cb(self, msg: Odometry):
        """
        Handle ground truth pose.
        Updates the visualizer's GT path.
        """
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        
        # ONLY initialize if we are explicitly waiting for it and haven't done it yet
        if self.wait_for_gt and not self.slam_initialized:
            # Extract yaw from quaternion
            q = msg.pose.pose.orientation
            siny_cosp = 2 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
            yaw = np.arctan2(siny_cosp, cosy_cosp)

            rospy.loginfo(f"SLAM: Initializing pose to x={x:.2f}, y={y:.2f}, theta={np.degrees(yaw):.2f}")
            
            self.slam.x[0, 0] = x
            self.slam.x[1, 0] = y
            self.slam.x[2, 0] = yaw
            
            self.slam_initialized = True
        
        # Update visualizer GT path
        self.visualizer.update_gt_pose(x, y)
        
        # Also maintain ROS path message for RViz
        current_time = rospy.Time.now()
        ps = PoseStamped()
        ps.header.stamp = current_time
        ps.header.frame_id = "map"
        ps.pose = msg.pose.pose
        
        self.gt_path.poses.append(ps)
        if len(self.gt_path.poses) > 1000:
            self.gt_path.poses = self.gt_path.poses[-1000:]
        
        self.gt_path.header.stamp = current_time
        self.gt_path_pub.publish(self.gt_path)

    def odom_cb(self, msg: Odometry):
        """Odometry callback for EKF prediction step."""
        if not self.slam_initialized:
            return
        
        v = msg.twist.twist.linear.x
        w = msg.twist.twist.angular.z
        
        # Store for odometry message
        self.last_v = v
        self.last_w = w

        t = msg.header.stamp.to_sec()
        if self.last_odom_time is None:
            self.last_odom_time = t
            rospy.loginfo("SLAM: First odometry message received")
            return

        dt = t - self.last_odom_time
        self.last_odom_time = t
        if dt <= 0.0:
            return

        # Debug: log velocities periodically
        self.odom_count += 1
        if self.odom_count % 50 == 0:
            rospy.loginfo(f"SLAM Predict: v={v:.4f} m/s, w={np.rad2deg(w):.2f} deg/s, dt={dt:.3f}s")
            rospy.loginfo(f"SLAM State: x={self.slam.x[0,0]:.3f}, y={self.slam.x[1,0]:.3f}, theta={np.rad2deg(self.slam.x[2,0]):.1f}deg")

        self.slam.predict(v, w, dt)
        self.publish_fast_state()

    def image_cb(self, msg: CompressedImage):
        """Image callback for AprilTag detection and EKF update."""
        if not self.slam_initialized:
            return
        if self.camera_params is None:
            rospy.logwarn_throttle(5.0, "Waiting for camera intrinsics...")
            return

        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception as e:
            rospy.logwarn("cv_bridge error: %s", e)
            return

        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

        detections = self.detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=self.camera_params,
            tag_size=self.tag_size,
        )

        # Draw detections on image for visualization
        debug_img = cv_img.copy()
        self.draw_detections(debug_img, detections)
        
        # Publish debug image
        self.publish_debug_image(debug_img, msg.header.stamp)

        if len(detections) == 0:
            return

        for det in detections:
            tag_id = det.tag_id

            # det.pose_t: 3x1 translation of tag in camera frame
            t = det.pose_t
            cam_x = float(t[0, 0])  # right
            cam_y = float(t[1, 0])  # down
            cam_z = float(t[2, 0])  # forward

            # Convert to 2D robot frame: x forward, y left
            dx = cam_z
            dy = -cam_x

            r = np.sqrt(dx**2 + dy**2)
            bearing = np.arctan2(dy, dx)

            z = np.array([r, bearing])
            self.slam.update(tag_id, z)

            # Log gating diagnostics (useful to see if tags are being rejected)
            if getattr(self.slam, 'last_update_accepted', None) is False:
                md2 = getattr(self.slam, 'last_update_mahal_dist_sq', None)
                gate = getattr(self.slam, 'mahal_gate', None)
                if md2 is not None and gate is not None:
                    rospy.logwarn_throttle(1.0, f"EKF update REJECTED tag {tag_id}: mahal^2={md2:.2f} > {gate:.2f}")
                else:
                    rospy.logwarn_throttle(1.0, f"EKF update REJECTED tag {tag_id}")
            elif getattr(self.slam, 'last_update_was_new_landmark', None) is True:
                rospy.loginfo_throttle(1.0, f"EKF initialized new landmark: tag {tag_id}")


            self.detection_count += 1
            self.last_detection_time = rospy.Time.now()

            rospy.loginfo_throttle(1.0, 
                f"Tag {tag_id} detected: range={r:.2f}m, bearing={np.rad2deg(bearing):.1f}deg")

        self.publish_fast_state()

    def draw_detections(self, img, detections):
        """Draw AprilTag detections on image for debugging."""
        for det in detections:
            # Draw the tag outline
            corners = det.corners.astype(int)
            
            # Draw polygon
            cv2.polylines(img, [corners], True, (0, 255, 0), 2)
            
            # Draw center
            center = det.center.astype(int)
            cv2.circle(img, tuple(center), 5, (0, 0, 255), -1)
            
            # Draw tag ID
            cv2.putText(img, f"ID: {det.tag_id}", 
                       (center[0] - 20, center[1] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Draw pose info if available
            if det.pose_t is not None:
                t = det.pose_t
                dist = np.sqrt(t[0]**2 + t[1]**2 + t[2]**2)
                cv2.putText(img, f"D: {float(dist):.2f}m",
                           (center[0] - 20, center[1] + 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

            # Draw corner numbers
            for i, corner in enumerate(corners):
                cv2.circle(img, tuple(corner), 3, (255, 0, 255), -1)
                cv2.putText(img, str(i), tuple(corner + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

        # Draw status info
        status_text = f"Detections: {len(detections)} | Total: {self.detection_count}"
        cv2.putText(img, status_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Draw landmark count
        lm_text = f"Landmarks in map: {len(self.slam.landmark_ids)}"
        cv2.putText(img, lm_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Draw robot pose estimate
        x, y, theta = self.slam.x[0, 0], self.slam.x[1, 0], self.slam.x[2, 0]
        pose_text = f"Pose: x={x:.2f}, y={y:.2f}, th={np.rad2deg(theta):.1f}deg"
        cv2.putText(img, pose_text, (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    def _publish_debug_image(self, img, stamp):
        try:
            msg = CompressedImage()
            msg.header.stamp = stamp
            msg.format = "jpeg"
            msg.data = cv2.imencode('.jpg', img)[1].tobytes()
            self.debug_img_pub.publish(msg)

            raw_msg = self.bridge.cv2_to_imgmsg(img, "bgr8")
            raw_msg.header.stamp = stamp
            self.debug_img_raw_pub.publish(raw_msg)
        except Exception as e:
            rospy.logwarn_throttle(5.0, f"Debug image error: {e}")
            
    def _viz_timer_cb(self, event):
        try:
            viz_img = self.visualizer.render(self.slam.x, self.slam.P, self.slam.landmark_ids)
            msg = CompressedImage()
            msg.header.stamp = rospy.Time.now()
            msg.format = "jpeg"
            msg.data = cv2.imencode('.jpg', cv2.cvtColor(viz_img, cv2.COLOR_RGB2BGR))[1].tobytes()
            self.viz_pub.publish(msg)
        except Exception as e:
            rospy.logwarn_throttle(5.0, f"Viz error: {e}")

    def _publish_state(self):
        self._publish_pose()
        self._publish_pose_cov()
        self._publish_odom()
        self._publish_path()
        self._publish_landmarks()
        self._publish_tf()

    def _publish_pose(self):
        x = self.slam.x
        yaw = float(x[2, 0])

        ps = PoseStamped()
        ps.header.stamp = rospy.Time.now()
        ps.header.frame_id = "map"
        ps.pose.position.x, ps.pose.position.y = float(x[0, 0]), float(x[1, 0])
        ps.pose.orientation.z, ps.pose.orientation.w = np.sin(yaw / 2), np.cos(yaw / 2)
        self.pose_pub.publish(ps)

    def _publish_pose_cov(self):
        x, P = self.slam.x, self.slam.P
        yaw = float(x[2, 0])

        msg = PoseWithCovarianceStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "map"
        msg.pose.pose.position.x, msg.pose.pose.position.y = float(x[0, 0]), float(x[1, 0])
        msg.pose.pose.orientation.z, msg.pose.pose.orientation.w = np.sin(yaw / 2), np.cos(yaw / 2)

        cov = np.zeros(36)
        cov[0], cov[1], cov[5] = P[0, 0], P[0, 1], P[0, 2]
        cov[6], cov[7], cov[11] = P[1, 0], P[1, 1], P[1, 2]
        cov[30], cov[31], cov[35] = P[2, 0], P[2, 1], P[2, 2]
        msg.pose.covariance = cov.tolist()
        self.pose_cov_pub.publish(msg)

    def _publish_odom(self):
        x = self.slam.x
        yaw = float(x[2, 0])

        msg = Odometry()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "map"
        msg.child_frame_id = "base_link"
        msg.pose.pose.position.x, msg.pose.pose.position.y = float(x[0, 0]), float(x[1, 0])
        msg.pose.pose.orientation.z, msg.pose.pose.orientation.w = np.sin(yaw / 2), np.cos(yaw / 2)
        msg.twist.twist.linear.x, msg.twist.twist.angular.z = self.last_v, self.last_w
        self.slam_odom_pub.publish(msg)

    def _publish_path(self):
        x = self.slam.x
        yaw = float(x[2, 0])

        ps = PoseStamped()
        ps.header.stamp = rospy.Time.now()
        ps.header.frame_id = "map"
        ps.pose.position.x, ps.pose.position.y = float(x[0, 0]), float(x[1, 0])
        ps.pose.orientation.z, ps.pose.orientation.w = np.sin(yaw / 2), np.cos(yaw / 2)

        self.path.poses.append(ps)
        if len(self.path.poses) > 1000:
            self.path.poses = self.path.poses[-1000:]
        self.path.header.stamp = ps.header.stamp
        self.path_pub.publish(self.path)

    def _publish_landmarks(self):
        ma = MarkerArray()

        # Points marker
        pts = Marker()
        pts.header.stamp = rospy.Time.now()
        pts.header.frame_id = "map"
        pts.ns, pts.id = "landmark_points", 0
        pts.type, pts.action = Marker.POINTS, Marker.ADD
        pts.scale.x = pts.scale.y = 0.08
        pts.color.r, pts.color.a = 1.0, 1.0

        for i, tag_id in enumerate(self.slam.landmark_ids):
            idx = 3 + 2 * i
            lx, ly = float(self.slam.x[idx, 0]), float(self.slam.x[idx + 1, 0])

            pts.points.append(Point(x=lx, y=ly, z=0))

            # Label
            txt = Marker()
            txt.header.stamp = rospy.Time.now()
            txt.header.frame_id = "map"
            txt.ns, txt.id = "landmark_labels", i + 100
            txt.type, txt.action = Marker.TEXT_VIEW_FACING, Marker.ADD
            txt.pose.position.x, txt.pose.position.y, txt.pose.position.z = lx, ly, 0.15
            txt.scale.z = 0.1
            txt.color.r = txt.color.g = txt.color.a = 1.0
            txt.text = f"Tag {tag_id}"
            ma.markers.append(txt)

            # Covariance ellipse
            ellipse = self._create_cov_ellipse(i, lx, ly)
            if ellipse:
                ma.markers.append(ellipse)

        ma.markers.append(pts)
        ma.markers.append(self._create_robot_marker())
        ma.markers.append(self._create_trajectory_marker())
        self.lm_pub.publish(ma)

    def _create_cov_ellipse(self, idx, lx, ly):
        lm_start = 3 + 2 * idx
        if lm_start + 1 >= self.slam.P.shape[0]:
            return None

        try:
            P_lm = self.slam.P[lm_start:lm_start + 2, lm_start:lm_start + 2]
            eigenvalues, eigenvectors = np.linalg.eigh(P_lm)
            scale = np.sqrt(5.991)

            m = Marker()
            m.header.stamp = rospy.Time.now()
            m.header.frame_id = "map"
            m.ns, m.id = "landmark_covariance", idx + 200
            m.type, m.action = Marker.CYLINDER, Marker.ADD
            m.pose.position.x, m.pose.position.y, m.pose.position.z = lx, ly, -0.01

            angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
            m.pose.orientation.z, m.pose.orientation.w = np.sin(angle / 2), np.cos(angle / 2)
            m.scale.x = scale * np.sqrt(max(eigenvalues[0], 1e-6)) * 2
            m.scale.y = scale * np.sqrt(max(eigenvalues[1], 1e-6)) * 2
            m.scale.z = 0.01
            m.color.r, m.color.g, m.color.a = 1.0, 0.5, 0.3
            return m
        except Exception:
            return None

    def _create_robot_marker(self):
        x = self.slam.x
        yaw = float(x[2, 0])

        m = Marker()
        m.header.stamp = rospy.Time.now()
        m.header.frame_id = "map"
        m.ns, m.id = "robot", 0
        m.type, m.action = Marker.ARROW, Marker.ADD
        m.pose.position.x, m.pose.position.y, m.pose.position.z = float(x[0, 0]), float(x[1, 0]), 0.02
        m.pose.orientation.z, m.pose.orientation.w = np.sin(yaw / 2), np.cos(yaw / 2)
        m.scale.x, m.scale.y, m.scale.z = 0.15, 0.05, 0.05
        m.color.g, m.color.a = 1.0, 1.0
        return m

    def _create_trajectory_marker(self):
        m = Marker()
        m.header.stamp = rospy.Time.now()
        m.header.frame_id = "map"
        m.ns, m.id = "trajectory", 0
        m.type, m.action = Marker.LINE_STRIP, Marker.ADD
        m.scale.x = 0.03
        m.color.g, m.color.b, m.color.a = 0.4, 1.0, 1.0

        for ps in self.path.poses:
            m.points.append(Point(x=float(ps.pose.position.x), y=float(ps.pose.position.y), z=0.01))
        return m

    def _publish_tf(self):
        x = self.slam.x
        yaw = float(x[2, 0])

        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = "map"
        t.child_frame_id = "base_link"
        t.transform.translation.x, t.transform.translation.y = float(x[0, 0]), float(x[1, 0])
        t.transform.rotation.z, t.transform.rotation.w = np.sin(yaw / 2), np.cos(yaw / 2)
        self.tf_broadcaster.sendTransform(t)


def main():
    rospy.init_node("apriltag_ekf_slam_node")
    AprilTagEkfSlamNode()
    rospy.spin()


if __name__ == "__main__":
    main()
