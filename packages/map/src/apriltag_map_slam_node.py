#!/usr/bin/env python3
import math
import numpy as np
import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped, Point
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import CompressedImage, CameraInfo, Image
from cv_bridge import CvBridge
import cv2
from std_srvs.srv import Trigger, TriggerResponse

from dt_apriltags import Detector
from map.include.map_slam_backend import MapSlam2D


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class AprilTagMapSlamNode:
    def __init__(self):
        rospy.init_node("apriltag_map_slam_node")

        self.image_topic = rospy.get_param("~image_topic", "/camera/compressed")
        self.odom_topic = rospy.get_param("~odom_topic", "/odom")
        self.camera_info_topic = rospy.get_param("~camera_info_topic", "/camera_node/camera_info")

        # ---- AprilTag params ----
        self.tag_size = float(rospy.get_param("~tag_size", 0.065))
        self.quad_decimate = float(rospy.get_param("~quad_decimate", 1.0))
        self.quad_sigma = float(rospy.get_param("~quad_sigma", 0.0))
        self.nthreads = int(rospy.get_param("~nthreads", 2))

        # ---- SLAM tuning ----
        Q_xy = float(rospy.get_param("~Q_xy", 0.15))
        Q_th = math.radians(float(rospy.get_param("~Q_theta_deg", 12.0)))
        R_rng = float(rospy.get_param("~R_range", 0.20))
        R_b = math.radians(float(rospy.get_param("~R_bearing_deg", 6.0)))
        huber_k = float(rospy.get_param("~huber_k", 1.5))
        gate_chi2 = float(rospy.get_param("~gate_chi2", 25.0))
        enable_gating = bool(rospy.get_param("~enable_gating", True))

        self.slam = MapSlam2D(
            odom_sigmas=(Q_xy, Q_xy, Q_th),
            meas_sigmas=(R_rng, R_b),
            huber_k=huber_k,
            enable_gating=enable_gating,
            gate_chi2=gate_chi2,
        )

        # ---- State ----
        self.prev_odom = None  # (x,y,theta) in odom frame
        self.camera_params = None  # [fx, fy, cx, cy]

        # ---- AprilTag detector ----
        self.detector = Detector(
            families="tag36h11",
            nthreads=self.nthreads,
            quad_decimate=self.quad_decimate,
            quad_sigma=self.quad_sigma,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0,
        )

        self.bridge = CvBridge()

        # ---- Publishers ----
        self.pub_odom = rospy.Publisher("slam_odom", Odometry, queue_size=10)
        self.pub_path = rospy.Publisher("slam_path", Path, queue_size=10)
        self.pub_pose = rospy.Publisher("slam_pose", PoseStamped, queue_size=10)
        self.pub_landmarks = rospy.Publisher("slam_landmarks", MarkerArray, queue_size=10)

        self.pub_dbg_raw = rospy.Publisher("slam_debug_image", Image, queue_size=1)
        self.pub_dbg_compressed = rospy.Publisher("slam_debug_image/compressed", CompressedImage, queue_size=1)

        # ---- TF ----
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        # ---- Subscribers ----
        self.sub_caminfo = rospy.Subscriber(self.camera_info_topic, CameraInfo, self.cb_caminfo, queue_size=1)
        self.sub_img = rospy.Subscriber(
            self.image_topic, CompressedImage, self.cb_image,
            queue_size=1, buff_size=2**24
        )
        self.sub_odom = rospy.Subscriber(self.odom_topic, Odometry, self.cb_odom, queue_size=50)

        # ---- Service ----
        self.srv_opt = rospy.Service("~optimize", Trigger, self.cb_optimize)

        # ---- Path message state ----
        self.path_msg = Path()
        self.path_msg.header.frame_id = "map"

        rospy.loginfo("AprilTag MAP-SLAM node started")
        rospy.loginfo(f"  image_topic: {self.image_topic}")
        rospy.loginfo(f"  camera_info_topic: {self.camera_info_topic}")
        rospy.loginfo(f"  odom_topic: {self.odom_topic}")
        rospy.loginfo(f"  tag_size: {self.tag_size} m, quad_decimate: {self.quad_decimate}")

    def cb_caminfo(self, msg: CameraInfo):
        K = msg.K
        fx = K[0]
        fy = K[4]
        cx = K[2]
        cy = K[5]
        self.camera_params = [fx, fy, cx, cy]
        rospy.loginfo(f"Got camera intrinsics fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
        # intrinsics are stable can unsubscribe
        try:
            self.sub_caminfo.unregister()
        except Exception:
            pass

    def cb_odom(self, msg: Odometry):
        # Extract planar pose from odom message
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation

        # yaw from quaternion
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        th = math.atan2(siny_cosp, cosy_cosp)

        if self.prev_odom is None:
            self.prev_odom = (x, y, th)
            return

        px, py, pth = self.prev_odom
        dx = x - px
        dy = y - py
        dth = wrap_angle(th - pth)

        # Rotate delta into previous heading frame (Pose2 local delta)
        c = math.cos(-pth)
        s = math.sin(-pth)
        dx_local = c * dx - s * dy
        dy_local = s * dx + c * dy

        self.slam.add_odometry(dx_local, dy_local, dth)
        self.prev_odom = (x, y, th)

        self.publish_all()

    def publish_debug_image(self, bgr, stamp):
        # Compressed
        try:
            msg = CompressedImage()
            msg.header.stamp = stamp
            msg.format = "jpeg"
            msg.data = np.array(cv2.imencode(".jpg", bgr)[1]).tobytes()
            self.pub_dbg_compressed.publish(msg)
        except Exception as e:
            rospy.logwarn_throttle(5.0, f"Error publishing compressed debug image: {e}")

        # Raw
        try:
            raw_msg = self.bridge.cv2_to_imgmsg(bgr, encoding="bgr8")
            raw_msg.header.stamp = stamp
            self.pub_dbg_raw.publish(raw_msg)
        except Exception as e:
            rospy.logwarn_throttle(5.0, f"Error publishing raw debug image: {e}")

    def cb_image(self, msg: CompressedImage):
        if self.camera_params is None:
            # This was the main issue in your current MAP node: it returns silently forever.
            rospy.logwarn_throttle(2.0, "Waiting for camera intrinsics (camera_info) â€” debug image not published yet.")
            return

        np_arr = np.frombuffer(msg.data, np.uint8)
        bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if bgr is None:
            rospy.logwarn_throttle(2.0, "cv2.imdecode failed for compressed image")
            return

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        detections = self.detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=self.camera_params,
            tag_size=float(self.tag_size),
        )

        dbg = bgr.copy()

        for det in detections:
            tag_id = int(det.tag_id)

            # det.pose_t is 3x1 translation in camera frame
            t = det.pose_t
            cam_x = float(t[0, 0])
            cam_z = float(t[2, 0])

            # Convention: forward=cam_z, left=-cam_x
            dx = cam_z
            dy = -cam_x

            rng = math.hypot(dx, dy)
            bearing = math.atan2(dy, dx)

            accepted, m2 = self.slam.add_tag_measurement(tag_id, rng, bearing)
            if not accepted:
                rospy.logwarn_throttle(1.0, f"MAP update REJECTED tag {tag_id}: mahal^2={m2:.2f} > {self.slam.gate_chi2:.2f}")

            # Debug overlay
            cxy = (int(det.center[0]), int(det.center[1]))
            cv2.circle(dbg, cxy, 4, (0, 255, 0), -1)
            cv2.putText(
                dbg,
                f"id={tag_id} r={rng:.2f} b={math.degrees(bearing):.1f}",
                (cxy[0] + 5, cxy[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

        # Publish debug image EVERY frame (even with 0 detections)
        self.publish_debug_image(dbg, msg.header.stamp)

    def cb_optimize(self, _req):
        iters = int(rospy.get_param("~batch_iters", 50))
        self.slam.batch_optimize(max_iters=iters)
        self.publish_all()
        return TriggerResponse(success=True, message=f"Optimized with {iters} iters")

    def create_trajectory_marker(self):
        poses = self.slam.get_all_poses()
        marker = Marker()
        marker.header.stamp = rospy.Time.now()
        marker.header.frame_id = "map"
        marker.ns = "trajectory"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.03

        marker.color.r = 0.0
        marker.color.g = 0.4
        marker.color.b = 1.0
        marker.color.a = 1.0
        marker.lifetime = rospy.Duration(0)

        marker.points = []
        for (x, y, _th) in poses:
            pt = Point()
            pt.x = float(x)
            pt.y = float(y)
            pt.z = 0.01
            marker.points.append(pt)
        return marker

    def create_landmark_markers(self):
        arr = MarkerArray()

        for (tag_id, x, y) in self.slam.get_all_landmarks():
            m = Marker()
            m.header.stamp = rospy.Time.now()
            m.header.frame_id = "map"
            m.ns = "landmarks"
            m.id = int(tag_id)
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(x)
            m.pose.position.y = float(y)
            m.pose.position.z = 0.0
            m.pose.orientation.w = 1.0
            m.scale.x = 0.08
            m.scale.y = 0.08
            m.scale.z = 0.08
            m.color.r = 1.0
            m.color.g = 0.6
            m.color.b = 0.0
            m.color.a = 1.0
            m.lifetime = rospy.Duration(0)
            arr.markers.append(m)

        arr.markers.append(self.create_trajectory_marker())
        return arr

    def publish_all(self):
        p = self.slam.last_pose()
        x, y, th = p.x(), p.y(), p.theta()

        now = rospy.Time.now()

        ps = PoseStamped()
        ps.header.stamp = now
        ps.header.frame_id = "map"
        ps.pose.position.x = x
        ps.pose.position.y = y
        ps.pose.orientation.z = math.sin(th / 2.0)
        ps.pose.orientation.w = math.cos(th / 2.0)
        self.pub_pose.publish(ps)

        od = Odometry()
        od.header = ps.header
        od.child_frame_id = "slam_base_link"
        od.pose.pose = ps.pose
        self.pub_odom.publish(od)

        self.path_msg.header.stamp = now
        self.path_msg.poses.append(ps)
        self.pub_path.publish(self.path_msg)

        self.pub_landmarks.publish(self.create_landmark_markers())

        t = TransformStamped()
        t.header.stamp = now
        t.header.frame_id = "map"
        t.child_frame_id = "slam_base_link"
        t.transform.translation.x = x
        t.transform.translation.y = y
        t.transform.translation.z = 0.0
        t.transform.rotation = ps.pose.orientation
        self.tf_broadcaster.sendTransform(t)

    def spin(self):
        rospy.spin()


if __name__ == "__main__":
    AprilTagMapSlamNode().spin()