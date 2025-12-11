#!/usr/bin/env python3
import rospy
import numpy as np
import cv2
import tf2_ros
from geometry_msgs.msg import TransformStamped

from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker, MarkerArray

from dt_apriltags import Detector
from map.include.graph_slam import LightweightGraphSlam2D


class LightweightMapSlamNode(object):
    def __init__(self):
        # Parameters
        self.veh = rospy.get_param("~veh", "")
        self.image_topic = rospy.get_param("~image_topic", "/camera/compressed")
        self.odom_topic = rospy.get_param("~odom_topic", "/odom")
        self.camera_info_topic = rospy.get_param("~camera_info_topic", "/camera_node/camera_info")
        self.tag_size = rospy.get_param("~tag_size", 0.065)
        
        # SLAM parameters
        self.window_size = rospy.get_param("~window_size", 50)
        self.keyframe_distance = rospy.get_param("~keyframe_distance", 0.15)
        self.keyframe_angle = rospy.get_param("~keyframe_angle", 15.0)
        self.optimization_rate = rospy.get_param("~optimization_rate", 1.0)
        
        self.camera_params = None
        self.camera_info_sub = rospy.Subscriber(
            self.camera_info_topic, CameraInfo, self.camera_info_cb, queue_size=1
        )

        # SLAM backend with covariance computation enabled
        self.slam = LightweightGraphSlam2D(
            window_size=self.window_size,
            keyframe_distance=self.keyframe_distance,
            keyframe_angle=np.deg2rad(self.keyframe_angle),
            compute_covariances=False  # Enable covariance extraction, estimation broken
        )
        
        self.last_odom_time = None

        # Path
        self.path = Path()
        self.path.header.frame_id = "map"

        # Velocity tracking
        self.last_v = 0.0
        self.last_w = 0.0


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
            queue_size=1, buff_size=2**24 # buffer size 16MB for compressed images
        )
        self.odom_sub = rospy.Subscriber(
            self.odom_topic, Odometry, self.odom_cb, queue_size=50 
        )

        # Publishers
        self.pose_pub = rospy.Publisher("slam_pose", PoseStamped, queue_size=10)
        self.lm_pub = rospy.Publisher("slam_landmarks", MarkerArray, queue_size=10)
        self.path_pub = rospy.Publisher("slam_path", Path, queue_size=10)
        self.slam_odom_pub = rospy.Publisher("slam_odom", Odometry, queue_size=10)
        
        # Debug image
        self.debug_img_pub = rospy.Publisher(
            "slam_debug_image/compressed", CompressedImage, queue_size=1
        )
        self.debug_img_raw_pub = rospy.Publisher(
            "slam_debug_image", Image, queue_size=1
        )

        # TF broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        # Timers
        self.optimization_timer = rospy.Timer(
            rospy.Duration(1.0 / self.optimization_rate), 
            self.optimization_callback
        )

        # Stats
        # self.detection_count = 0
        self.odom_count = 0

        rospy.loginfo("=== MAP-SLAM Node (EKF-matched detection settings) ===")
        rospy.loginfo(f"  Detector: quad_decimate=2.0, nthreads=1 (same as EKF)")
        rospy.loginfo(f"  Image: full resolution, every frame (same as EKF)")
        rospy.loginfo(f"  Window size: {self.window_size}")
        rospy.loginfo(f"  Keyframe: {self.keyframe_distance}m / {self.keyframe_angle}deg")

    def camera_info_cb(self, msg: CameraInfo):
        K = msg.K
        # NO decimation - use original camera params
        self.camera_params = [K[0], K[4], K[2], K[5]]
        rospy.loginfo(f"Camera: fx={K[0]:.1f}, fy={K[4]:.1f}, cx={K[2]:.1f}, cy={K[5]:.1f}")
        self.camera_info_sub.unregister()

    def odom_cb(self, msg: Odometry):
        v = msg.twist.twist.linear.x
        w = msg.twist.twist.angular.z
        self.last_v = v
        self.last_w = w

        t = msg.header.stamp.to_sec()
        if self.last_odom_time is None:
            self.last_odom_time = t
            rospy.loginfo("First odometry received")
            return

        dt = t - self.last_odom_time
        self.last_odom_time = t
        
        if dt <= 0.0 or dt > 1.0:
            return

        self.slam.add_odometry_constraint(v, w, dt)
        
        self.odom_count += 1
        if self.odom_count % 50 == 0:
            pose = self.slam.get_current_pose()
            rospy.loginfo(f"Odom #{self.odom_count}: pose=({pose[0]:.2f}, {pose[1]:.2f}, {np.rad2deg(pose[2]):.1f}deg)")

    def image_cb(self, msg: CompressedImage):
        """Process EVERY frame at FULL resolution (like EKF)."""
        if self.camera_params is None:
            rospy.logwarn_throttle(5.0, "Waiting for camera intrinsics...")
            return

        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception as e:
            rospy.logwarn(f"Image decode error: {e}")
            return

        # NO DECIMATION - full resolution
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

        detections = self.detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=self.camera_params,
            tag_size=self.tag_size,
        )

        # Draw and publish debug image
        debug_img = cv_img.copy()
        self.draw_detections(debug_img, detections)
        self.publish_debug_image(debug_img, msg.header.stamp)

        if len(detections) == 0:
            return

        for det in detections:
            tag_id = det.tag_id
            t = det.pose_t
            
            cam_x = float(t[0, 0])
            cam_y = float(t[1, 0])
            cam_z = float(t[2, 0])

            # Convert to robot frame
            dx = cam_z
            dy = -cam_x
            r = np.sqrt(dx**2 + dy**2)
            bearing = np.arctan2(dy, dx)

            self.slam.add_landmark_observation(tag_id, r, bearing)
            # self.detection_count += 1

            rospy.loginfo_throttle(1.0, 
                f"Tag {tag_id}: range={r:.2f}m, bearing={np.rad2deg(bearing):.1f}deg")

        self.publish_all()

    def draw_detections(self, img, detections):
        """Draw detections (same style as EKF)."""
        for det in detections:
            corners = det.corners.astype(int)
            cv2.polylines(img, [corners], True, (0, 255, 0), 2)
            
            center = det.center.astype(int)
            cv2.circle(img, tuple(center), 5, (0, 0, 255), -1)
            
            cv2.putText(img, f"ID: {det.tag_id}", 
                       (center[0] - 20, center[1] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            if det.pose_t is not None:
                t = det.pose_t
                dist = np.sqrt(t[0]**2 + t[1]**2 + t[2]**2)
                cv2.putText(img, f"D: {float(dist):.2f}m",
                           (center[0] - 20, center[1] + 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

            for i, corner in enumerate(corners):
                cv2.circle(img, tuple(corner), 3, (255, 0, 255), -1)
                cv2.putText(img, str(i), tuple(corner + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

        # Status overlay
        status_text = f"Detections: {len(detections)} | Total: N/A"
        cv2.putText(img, status_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        lm_text = f"Landmarks: {len(self.slam.landmarks)}"
        cv2.putText(img, lm_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        pose = self.slam.get_current_pose()
        pose_text = f"Pose: x={pose[0]:.2f}, y={pose[1]:.2f}, th={np.rad2deg(pose[2]):.1f}deg"
        cv2.putText(img, pose_text, (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    def publish_debug_image(self, img, stamp):
        """Publish both raw and compressed (same as EKF)."""
        try:
            msg = CompressedImage()
            msg.header.stamp = stamp
            msg.format = "jpeg"
            msg.data = np.array(cv2.imencode('.jpg', img)[1]).tobytes()
            self.debug_img_pub.publish(msg)
        except Exception as e:
            rospy.logwarn_throttle(5.0, f"Error publishing compressed: {e}")

        try:
            raw_msg = self.bridge.cv2_to_imgmsg(img, "bgr8")
            raw_msg.header.stamp = stamp
            self.debug_img_raw_pub.publish(raw_msg)
        except Exception as e:
            rospy.logwarn_throttle(5.0, f"Error publishing raw: {e}")

    def optimization_callback(self, event=None):
        if len(self.slam.keyframe_poses) > 1:
            self.slam.optimize()
            self.publish_all()

    def publish_all(self):
        self.publish_pose()
        self.publish_slam_odometry()
        self.publish_landmarks()
        self.publish_path()
        self.publish_tf()

    def publish_pose(self):
        pose = self.slam.get_current_pose()
        
        ps = PoseStamped()
        ps.header.stamp = rospy.Time.now()
        ps.header.frame_id = "map"
        ps.pose.position.x = float(pose[0])
        ps.pose.position.y = float(pose[1])
        ps.pose.position.z = 0.0

        yaw = float(pose[2])
        ps.pose.orientation.z = np.sin(yaw / 2.0)
        ps.pose.orientation.w = np.cos(yaw / 2.0)

        self.pose_pub.publish(ps)

    def publish_slam_odometry(self):
        pose = self.slam.get_current_pose()
        
        odom = Odometry()
        odom.header.stamp = rospy.Time.now()
        odom.header.frame_id = "map"
        odom.child_frame_id = "base_link"

        odom.pose.pose.position.x = float(pose[0])
        odom.pose.pose.position.y = float(pose[1])
        odom.pose.pose.position.z = 0.0

        yaw = float(pose[2])
        odom.pose.pose.orientation.z = np.sin(yaw / 2.0)
        odom.pose.pose.orientation.w = np.cos(yaw / 2.0)

        odom.twist.twist.linear.x = self.last_v
        odom.twist.twist.angular.z = self.last_w

        self.slam_odom_pub.publish(odom)

    def publish_landmarks(self):
        ma = MarkerArray()
        
        landmarks = self.slam.get_all_landmarks()
        covariances = self.slam.get_all_landmark_covariances()
        
        # Points marker
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

        for i, (tag_id, pos) in enumerate(landmarks.items()):
            lx, ly = float(pos[0]), float(pos[1])
            
            # Add point
            p = Point()
            p.x = lx
            p.y = ly
            p.z = 0.0
            points_marker.points.append(p)

            # Text label
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

            # Covariance ellipse (like EKF)
            if tag_id in covariances:
                ellipse = self.create_covariance_ellipse(i, lx, ly, covariances[tag_id])
                if ellipse is not None:
                    ma.markers.append(ellipse)

        ma.markers.append(points_marker)
        
        # Robot marker
        robot_marker = self.create_robot_marker()
        ma.markers.append(robot_marker)

        self.lm_pub.publish(ma)

    def create_covariance_ellipse(self, idx, lx, ly, P_lm):
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(P_lm)
            # 95% confidence ellipse
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
            
            angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
            marker.pose.orientation.z = np.sin(angle / 2.0)
            marker.pose.orientation.w = np.cos(angle / 2.0)
            
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

    def create_robot_marker(self):
        pose = self.slam.get_current_pose()
        
        marker = Marker()
        marker.header.stamp = rospy.Time.now()
        marker.header.frame_id = "map"
        marker.ns = "robot"
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        
        marker.pose.position.x = float(pose[0])
        marker.pose.position.y = float(pose[1])
        marker.pose.position.z = 0.02
        
        yaw = float(pose[2])
        marker.pose.orientation.z = np.sin(yaw / 2.0)
        marker.pose.orientation.w = np.cos(yaw / 2.0)
        
        marker.scale.x = 0.15
        marker.scale.y = 0.05
        marker.scale.z = 0.05
        
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        marker.lifetime = rospy.Duration(0)
        
        return marker

    def publish_path(self):
        path = Path()
        path.header.stamp = rospy.Time.now()
        path.header.frame_id = "map"
        
        for pose_arr in self.slam.get_all_poses():
            ps = PoseStamped()
            ps.header.frame_id = "map"
            ps.pose.position.x = float(pose_arr[0])
            ps.pose.position.y = float(pose_arr[1])
            
            yaw = float(pose_arr[2])
            ps.pose.orientation.z = np.sin(yaw / 2.0)
            ps.pose.orientation.w = np.cos(yaw / 2.0)
            
            path.poses.append(ps)
        
        self.path_pub.publish(path)

    def publish_tf(self):
        pose = self.slam.get_current_pose()

        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = "map"
        t.child_frame_id = "base_link"

        t.transform.translation.x = float(pose[0])
        t.transform.translation.y = float(pose[1])
        t.transform.translation.z = 0.0

        yaw = float(pose[2])
        t.transform.rotation.z = np.sin(yaw / 2.0)
        t.transform.rotation.w = np.cos(yaw / 2.0)

        self.tf_broadcaster.sendTransform(t)


def main():
    rospy.init_node("lightweight_map_slam_node")
    node = LightweightMapSlamNode()
    rospy.spin()


if __name__ == "__main__":
    main()