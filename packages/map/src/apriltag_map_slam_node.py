#!/usr/bin/env python3
import rospy
import numpy as np
import cv2
import tf2_ros
from geometry_msgs.msg import TransformStamped, Pose

from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped, Point, PoseWithCovarianceStamped
from visualization_msgs.msg import Marker, MarkerArray

from dt_apriltags import Detector
from slam.include.slam.map_slam import MapSlam2D


class AprilTagMapSlamNode(object):
    def __init__(self):
        # Parameters (can be overridden with rosparam)
        self.veh = rospy.get_param("~veh", "")
        self.image_topic = rospy.get_param("~image_topic", "/camera/compressed")
        self.odom_topic = rospy.get_param("~odom_topic", "/odom")
        self.gt_topic = rospy.get_param("~gt_topic", "/ground_truth/odom")  # Ground truth topic (Pose type)
        self.use_gt = rospy.get_param("~use_ground_truth", True)
        self.camera_params = None
        self.camera_info_topic = rospy.get_param(
            "~camera_info_topic", "/camera_node/camera_info"
        )

        self.camera_info_sub = rospy.Subscriber(
            self.camera_info_topic, CameraInfo, self.camera_info_cb, queue_size=1
        )

        # Tag size in meters
        self.tag_size = rospy.get_param("~tag_size", 0.065)

        # Backend SLAM filter
        self.slam = MapSlam2D()
        self.last_odom_time = None

        # Path history for visualization
        self.path = Path()
        self.path.header.frame_id = "map"
        
        # Ground truth path history
        self.gt_path = Path()
        self.gt_path.header.frame_id = "map"

        # Store last velocities for odometry message
        self.last_v = 0.0
        self.last_w = 0.0
        
        # Ground truth tracking for velocity estimation
        self.last_gt_pose = None
        self.last_gt_time = None

        # AprilTag detector (tag36h11 is what Duckietown uses by default)
        self.detector = Detector(
            families='tag36h11',
            nthreads=1,
            quad_decimate=2.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0,
        )

        # ROS I/O
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(
            self.image_topic, CompressedImage, self.image_cb,
            queue_size=1, buff_size=2**24
        )
        self.odom_sub = rospy.Subscriber(
            self.odom_topic, Odometry, self.odom_cb, queue_size=50
        )
        
        # Ground truth subscriber - accepts Pose messages (from gt_pose_visualizer_node)
        if self.use_gt:
            self.gt_sub = rospy.Subscriber(
                self.gt_topic, Odometry, self.gt_pose_cb, queue_size=10
            )

        # Publishers
        self.pose_pub = rospy.Publisher("slam_pose", PoseStamped, queue_size=10)
        self.pose_cov_pub = rospy.Publisher("slam_pose_cov", PoseWithCovarianceStamped, queue_size=10)
        self.lm_pub = rospy.Publisher("slam_landmarks", MarkerArray, queue_size=10)
        self.path_pub = rospy.Publisher("slam_path", Path, queue_size=10)
        
        # Odometry publisher for SLAM estimate (for RVIZ Axes visualization)
        self.slam_odom_pub = rospy.Publisher("slam_odom", Odometry, queue_size=10)

        """         
        # Odometry publisher for ground truth (for RVIZ Axes visualization)
        self.gt_odom_pub = rospy.Publisher("gt_odom", Odometry, queue_size=10)
        """ 

        # Ground truth path publisher
        self.gt_path_pub = rospy.Publisher("gt_path", Path, queue_size=10)
        
        # Debug image publisher - shows detected AprilTags
        self.debug_img_pub = rospy.Publisher("slam_debug_image/compressed", CompressedImage, queue_size=1)
        self.debug_img_raw_pub = rospy.Publisher("slam_debug_image", Image, queue_size=1)

        # TF broadcaster for RViz visualization
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        # Stats for logging
        self.detection_count = 0
        self.last_detection_time = None
        self.odom_count = 0

        rospy.loginfo("AprilTag MAP-SLAM node initialized")
        rospy.loginfo(f"  Image topic: {self.image_topic}")
        rospy.loginfo(f"  Odom topic: {self.odom_topic}")
        rospy.loginfo(f"  GT topic: {self.gt_topic}")
        rospy.loginfo(f"  Tag size: {self.tag_size}m")

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

    # ---------- GROUND TRUTH CALLBACK (Pose message) ----------
    def gt_pose_cb(self, msg: Odometry):
        """
        Handle ground truth pose from gt_pose_visualizer_node.
        Converts Pose to Odometry for RVIZ visualization.
        """
        current_time = rospy.Time.now()
        
        # Convert Pose to Odometry for RVIZ
        gt_odom = Odometry()
        gt_odom.header.stamp = current_time
        gt_odom.header.frame_id = "map"
        gt_odom.child_frame_id = "base_link_gt"
        
        gt_odom.pose.pose = msg
        
        # Estimate velocity from pose changes (optional, for twist field)
        if self.last_gt_pose is not None and self.last_gt_time is not None:
            dt = (current_time - self.last_gt_time).to_sec()
            if dt > 0.001:
                dx = msg.position.x - self.last_gt_pose.position.x
                dy = msg.position.y - self.last_gt_pose.position.y
                
                # Get yaw from quaternion
                qz = msg.orientation.z
                qw = msg.orientation.w
                yaw = 2.0 * np.arctan2(qz, qw)
                
                qz_prev = self.last_gt_pose.orientation.z
                qw_prev = self.last_gt_pose.orientation.w
                yaw_prev = 2.0 * np.arctan2(qz_prev, qw_prev)
                
                # Linear velocity in robot frame
                dist = np.sqrt(dx**2 + dy**2)
                gt_odom.twist.twist.linear.x = dist / dt
                
                # Angular velocity
                dyaw = yaw - yaw_prev
                # Wrap angle
                while dyaw > np.pi:
                    dyaw -= 2*np.pi
                while dyaw < -np.pi:
                    dyaw += 2*np.pi
                gt_odom.twist.twist.angular.z = dyaw / dt
        
        self.last_gt_pose = msg
        self.last_gt_time = current_time
        
        self.gt_odom_pub.publish(gt_odom)
        
        # Add to ground truth path
        ps = PoseStamped()
        ps.header.stamp = current_time
        ps.header.frame_id = "map"
        ps.pose = msg
        
        self.gt_path.poses.append(ps)
        if len(self.gt_path.poses) > 1000:
            self.gt_path.poses = self.gt_path.poses[-1000:]
        
        self.gt_path.header.stamp = current_time
        self.gt_path_pub.publish(self.gt_path)

    # ---------- ODOMETRY -> PREDICT ----------
    def odom_cb(self, msg: Odometry): # TODO
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
        self.publish_all()

    # ---------- CAMERA IMAGE -> APRILTAG DETECTION -> UPDATE ----------
    def image_cb(self, msg: CompressedImage): # TODO
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

            self.detection_count += 1
            self.last_detection_time = rospy.Time.now()

            rospy.loginfo_throttle(1.0, 
                f"Tag {tag_id} detected: range={r:.2f}m, bearing={np.rad2deg(bearing):.1f}deg")

        self.publish_all()

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

    def publish_debug_image(self, img, stamp):
        """Publish the debug image with detections drawn."""
        # Publish as compressed
        try:
            msg = CompressedImage()
            msg.header.stamp = stamp
            msg.format = "jpeg"
            msg.data = np.array(cv2.imencode('.jpg', img)[1]).tobytes()
            self.debug_img_pub.publish(msg)
        except Exception as e:
            rospy.logwarn_throttle(5.0, f"Error publishing compressed debug image: {e}")

        # Also publish raw for rqt_image_view
        try:
            raw_msg = self.bridge.cv2_to_imgmsg(img, "bgr8")
            raw_msg.header.stamp = stamp
            self.debug_img_raw_pub.publish(raw_msg)
        except Exception as e:
            rospy.logwarn_throttle(5.0, f"Error publishing raw debug image: {e}")

    # ---------- PUBLISH ----------
    def publish_all(self):
        """Publish all visualization data."""
        self.publish_pose()
        self.publish_pose_with_covariance()
        self.publish_slam_odometry()
        self.publish_landmarks()
        self.publish_path()
        self.publish_tf()

    def publish_pose(self): # TODO
        ps = PoseStamped()
        ps.header.stamp = rospy.Time.now()
        ps.header.frame_id = "map"

        x = self.slam.x
        ps.pose.position.x = float(x[0, 0])
        ps.pose.position.y = float(x[1, 0])
        ps.pose.position.z = 0.0

        yaw = float(x[2, 0])
        qz = np.sin(yaw / 2.0)
        qw = np.cos(yaw / 2.0)
        ps.pose.orientation.z = qz
        ps.pose.orientation.w = qw

        self.pose_pub.publish(ps)

    def publish_slam_odometry(self): # TODO
        """
        Publish SLAM estimated pose as Odometry message for RVIZ visualization.
        This allows using the rviz/Odometry display with Axes shape.
        """
        odom = Odometry()
        odom.header.stamp = rospy.Time.now()
        odom.header.frame_id = "map"
        odom.child_frame_id = "base_link"

        x = self.slam.x
        odom.pose.pose.position.x = float(x[0, 0])
        odom.pose.pose.position.y = float(x[1, 0])
        odom.pose.pose.position.z = 0.0

        yaw = float(x[2, 0])
        odom.pose.pose.orientation.x = 0.0
        odom.pose.pose.orientation.y = 0.0
        odom.pose.pose.orientation.z = np.sin(yaw / 2.0)
        odom.pose.pose.orientation.w = np.cos(yaw / 2.0)

        # Add covariance from Map (6x6 row-major)
        # ROS covariance order: x, y, z, roll, pitch, yaw
        cov = np.zeros(36)
        P = self.slam.P
        cov[0] = P[0, 0]   # xx
        cov[1] = P[0, 1]   # xy
        cov[5] = P[0, 2]   # x-yaw
        cov[6] = P[1, 0]   # yx
        cov[7] = P[1, 1]   # yy
        cov[11] = P[1, 2]  # y-yaw
        cov[30] = P[2, 0]  # yaw-x
        cov[31] = P[2, 1]  # yaw-y
        cov[35] = P[2, 2]  # yaw-yaw
        odom.pose.covariance = cov.tolist()

        # Add velocity information
        odom.twist.twist.linear.x = self.last_v
        odom.twist.twist.linear.y = 0.0
        odom.twist.twist.linear.z = 0.0
        odom.twist.twist.angular.x = 0.0
        odom.twist.twist.angular.y = 0.0
        odom.twist.twist.angular.z = self.last_w

        self.slam_odom_pub.publish(odom)

    def publish_pose_with_covariance(self): # TODO
        """Publish pose with covariance for visualization in RViz."""
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "map"

        x = self.slam.x
        msg.pose.pose.position.x = float(x[0, 0])
        msg.pose.pose.position.y = float(x[1, 0])
        msg.pose.pose.position.z = 0.0

        yaw = float(x[2, 0])
        msg.pose.pose.orientation.z = np.sin(yaw / 2.0)
        msg.pose.pose.orientation.w = np.cos(yaw / 2.0)

        # Fill covariance (6x6 row-major, we only have x, y, theta)
        # ROS covariance order: x, y, z, roll, pitch, yaw
        cov = np.zeros(36)
        P = self.slam.P
        cov[0] = P[0, 0]   # xx
        cov[1] = P[0, 1]   # xy
        cov[6] = P[1, 0]   # yx
        cov[7] = P[1, 1]   # yy
        cov[35] = P[2, 2]  # yaw-yaw
        msg.pose.covariance = cov.tolist()

        self.pose_cov_pub.publish(msg)

    def publish_landmarks(self): # TODO
        """Publish landmark markers for RViz visualization."""
        ma = MarkerArray()
        
        # Points marker for all landmarks
        points_marker = Marker()
        points_marker.header.stamp = rospy.Time.now()
        points_marker.header.frame_id = "map"
        points_marker.ns = "landmark_points"
        points_marker.id = 0
        points_marker.type = Marker.POINTS
        points_marker.action = Marker.ADD
        points_marker.scale.x = 0.08
        points_marker.scale.y = 0.08
        points_marker.color.r = 1.0
        points_marker.color.g = 0.0
        points_marker.color.b = 0.0
        points_marker.color.a = 1.0
        points_marker.lifetime = rospy.Duration(0)

        # Text markers for landmark IDs
        for i, tag_id in enumerate(self.slam.landmark_ids):
            lm_start = 3 + 2 * i
            lx = float(self.slam.x[lm_start, 0])
            ly = float(self.slam.x[lm_start + 1, 0])

            # Add point
            p = Point()
            p.x = lx
            p.y = ly
            p.z = 0.0
            points_marker.points.append(p)

            # Add text label
            text_marker = Marker()
            text_marker.header.stamp = rospy.Time.now()
            text_marker.header.frame_id = "map"
            text_marker.ns = "landmark_labels"
            text_marker.id = i + 100
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.pose.position.x = lx
            text_marker.pose.position.y = ly
            text_marker.pose.position.z = 0.15
            text_marker.scale.z = 0.1
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 0.0
            text_marker.color.a = 1.0
            text_marker.text = f"Tag {tag_id}"
            text_marker.lifetime = rospy.Duration(0)
            ma.markers.append(text_marker)

            # Add covariance ellipse
            ellipse_marker = self.create_covariance_ellipse(i, lx, ly)
            if ellipse_marker is not None:
                ma.markers.append(ellipse_marker)

        ma.markers.append(points_marker)
        
        # Add robot marker
        robot_marker = self.create_robot_marker()
        ma.markers.append(robot_marker)

        self.lm_pub.publish(ma)

    def create_covariance_ellipse(self, idx, lx, ly): # TODO
        """Create a marker showing the covariance ellipse for a landmark."""
        lm_start = 3 + 2 * idx
        if lm_start + 1 >= self.slam.P.shape[0]:
            return None

        # Extract 2x2 covariance for this landmark
        P_lm = self.slam.P[lm_start:lm_start+2, lm_start:lm_start+2]
        
        # Compute eigenvalues/eigenvectors for ellipse
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(P_lm)
            # 95% confidence ellipse (chi-squared with 2 DOF, 95% = 5.991)
            scale = np.sqrt(5.991)
            
            marker = Marker()
            marker.header.stamp = rospy.Time.now()
            marker.header.frame_id = "map"
            marker.ns = "landmark_covariance"
            marker.id = idx + 200
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            marker.pose.position.x = lx
            marker.pose.position.y = ly
            marker.pose.position.z = -0.01
            
            # Orientation from eigenvector
            angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
            marker.pose.orientation.z = np.sin(angle / 2.0)
            marker.pose.orientation.w = np.cos(angle / 2.0)
            
            # Scale from eigenvalues
            marker.scale.x = scale * np.sqrt(max(eigenvalues[0], 1e-6)) * 2
            marker.scale.y = scale * np.sqrt(max(eigenvalues[1], 1e-6)) * 2
            marker.scale.z = 0.01
            
            marker.color.r = 1.0
            marker.color.g = 0.5
            marker.color.b = 0.0
            marker.color.a = 0.3
            marker.lifetime = rospy.Duration(0)
            
            return marker
        except Exception:
            return None

    def create_robot_marker(self): # TODO
        """Create an arrow marker showing the robot pose."""
        marker = Marker()
        marker.header.stamp = rospy.Time.now()
        marker.header.frame_id = "map"
        marker.ns = "robot"
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        
        x = self.slam.x
        marker.pose.position.x = float(x[0, 0])
        marker.pose.position.y = float(x[1, 0])
        marker.pose.position.z = 0.02
        
        yaw = float(x[2, 0])
        marker.pose.orientation.z = np.sin(yaw / 2.0)
        marker.pose.orientation.w = np.cos(yaw / 2.0)
        
        marker.scale.x = 0.15  # length
        marker.scale.y = 0.05  # width
        marker.scale.z = 0.05  # height
        
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        marker.lifetime = rospy.Duration(0)
        
        return marker

    def publish_path(self): # TODO
        """Publish the robot's path history."""
        x = self.slam.x
        
        ps = PoseStamped()
        ps.header.stamp = rospy.Time.now()
        ps.header.frame_id = "map"
        ps.pose.position.x = float(x[0, 0])
        ps.pose.position.y = float(x[1, 0])
        ps.pose.position.z = 0.0
        
        yaw = float(x[2, 0])
        ps.pose.orientation.z = np.sin(yaw / 2.0)
        ps.pose.orientation.w = np.cos(yaw / 2.0)
        
        # Add to path (limit to last 1000 poses)
        self.path.poses.append(ps)
        if len(self.path.poses) > 1000:
            self.path.poses = self.path.poses[-1000:]
        
        self.path.header.stamp = rospy.Time.now()
        self.path_pub.publish(self.path)

    def publish_tf(self): # TODO
        """Publish TF transform from map to base_link for RViz visualization."""
        x = self.slam.x

        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = "map"
        t.child_frame_id = "base_link"

        t.transform.translation.x = float(x[0, 0])
        t.transform.translation.y = float(x[1, 0])
        t.transform.translation.z = 0.0

        yaw = float(x[2, 0])
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = np.sin(yaw / 2.0)
        t.transform.rotation.w = np.cos(yaw / 2.0)

        self.tf_broadcaster.sendTransform(t)


def main():
    rospy.init_node("apriltag_map_slam_node")
    node = AprilTagMapSlamNode()
    rospy.spin()


if __name__ == "__main__":
    main()