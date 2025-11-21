#!/usr/bin/env python3
import rospy
import numpy as np
import cv2

from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker, MarkerArray

from dt_apriltags import Detector
from packages.slam.include.slam.ekf_slam import EkfSlam2D


class AprilTagEkfSlamNode(object):
    def __init__(self):
        # Parameters (can be overridden with rosparam)
        self.image_topic = rospy.get_param("~image_topic", "/camera/image_raw")
        self.odom_topic  = rospy.get_param("~odom_topic",  "/odom")

        # Camera intrinsics from your calibration
        fx = rospy.get_param("~fx")
        fy = rospy.get_param("~fy")
        cx = rospy.get_param("~cx")
        cy = rospy.get_param("~cy")
        self.camera_params = [fx, fy, cx, cy]

        # Tag size in meters
        self.tag_size = rospy.get_param("~tag_size", 0.065)

        # Backend SLAM filter
        self.slam = EkfSlam2D()
        self.last_odom_time = None

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

        self.pose_pub = rospy.Publisher("slam_pose", PoseStamped, queue_size=10)
        self.lm_pub   = rospy.Publisher("slam_landmarks", MarkerArray, queue_size=10)
        
        self.camera_params = None
        self.camera_info_topic = rospy.get_param(
            "~camera_info_topic", "/camera_node/camera_info"
        )

        self.camera_info_sub = rospy.Subscriber(
            self.camera_info_topic, CameraInfo, self.camera_info_cb, queue_size=1
        )

        rospy.loginfo("AprilTag EKF-SLAM node initialized")

    def camera_info_cb(self, msg: CameraInfo):
        # K is row-major: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
        K = msg.K
        fx = K[0]
        fy = K[4]
        cx = K[2]
        cy = K[5]
        self.camera_params = [fx, fy, cx, cy]

        rospy.loginfo_once(f"Got camera intrinsics fx={fx}, fy={fy}, cx={cx}, cy={cy}")
        self.camera_info_sub.unregister()

    # ---------- ODOMETRY -> PREDICT ----------
    def odom_cb(self, msg: Odometry):
        v = msg.twist.twist.linear.x
        w = msg.twist.twist.angular.z

        t = msg.header.stamp.to_sec()
        if self.last_odom_time is None:
            self.last_odom_time = t
            return

        dt = t - self.last_odom_time
        self.last_odom_time = t
        if dt <= 0.0:
            return

        self.slam.predict(v, w, dt)
        self.publish_pose()
        self.publish_landmarks()

    # ---------- CAMERA IMAGE -> APRILTAG DETECTION -> UPDATE ----------
    def image_cb(self, msg: CompressedImage):
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

        self.publish_pose()
        self.publish_landmarks()

    # ---------- PUBLISH ----------
    def publish_pose(self):
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

    def publish_landmarks(self):
        ma = MarkerArray()
        marker = Marker()
        marker.header.stamp = rospy.Time.now()
        marker.header.frame_id = "map"
        marker.ns = "landmarks"
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.scale.x = 0.05
        marker.scale.y = 0.05
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        for i, tag_id in enumerate(self.slam.landmark_ids):
            lm_start = 3 + 2 * i
            lx = float(self.slam.x[lm_start, 0])
            ly = float(self.slam.x[lm_start + 1, 0])

            p = Point()
            p.x = lx
            p.y = ly
            p.z = 0.0
            marker.points.append(p)

        ma.markers.append(marker)
        self.lm_pub.publish(ma)


def main():
    rospy.init_node("apriltag_ekf_slam_node")
    node = AprilTagEkfSlamNode()
    rospy.spin()


if __name__ == "__main__":
    main()
