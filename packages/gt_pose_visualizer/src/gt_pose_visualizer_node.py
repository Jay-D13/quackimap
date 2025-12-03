#!/usr/bin/env python3

import rospy
import numpy as np
from geometry_msgs.msg import Pose
from nav_msgs.msg import Odometry
from dt_robot_utils import get_robot_name
from duckietown.sdk.robots.duckiebot import DB21J
from duckietown.dtros import DTROS, NodeType, TopicType


class GTPoseVisualizerNode(DTROS):

    def __init__(self):
        super(GTPoseVisualizerNode, self).__init__(node_name="gt_pose_visualizer_node", node_type=NodeType.DRIVER)
        self._robot_name = get_robot_name()

        # Here we use the Duckietown SDK to connect directly to the entity in the duckiematrix to get the
        # robot pose. Pointing to 'state' node because pose is located at .../state/pose
        self.robot: DB21J = DB21J("map_0/vehicle_0/state", simulated=True)
        self.robot.pose.start()

        # Previous pose for velocity estimation
        self.last_pose = None
        self.last_time = None

        # create publishers
        self._odom_pub = rospy.Publisher(
            "~gt_odom",
            Odometry,
            queue_size=1,
            dt_topic_type=TopicType.DRIVER,
            dt_help="The ground truth odometry of the robot from the Duckiematrix",
        )

        rospy.Timer(rospy.Duration(0.1), self.publish_pose)
        
        rospy.loginfo("GT Pose Visualizer initialized")

    def publish_pose(self, event=None):

        pose = self.robot.pose.capture()

        if pose is None:
            return

        current_time = rospy.Time.now()
        
        # Build Pose message
        pose_msg = Pose()
        pose_msg.position.x = pose["position"]["x"]
        pose_msg.position.y = pose["position"]["y"]
        pose_msg.position.z = pose["position"]["z"]
        pose_msg.orientation.x = pose["rotation"]["x"]
        pose_msg.orientation.y = pose["rotation"]["y"]
        pose_msg.orientation.z = pose["rotation"]["z"]
        pose_msg.orientation.w = pose["rotation"]["w"]
        
        # self._pub.publish(pose_msg)
        
        odom_msg = Odometry()
        odom_msg.header.stamp = current_time
        odom_msg.header.frame_id = "map"
        odom_msg.child_frame_id = "base_link_gt"
        
        odom_msg.pose.pose = pose_msg
        
        # Estimate velocities from pose difference
        if self.last_pose is not None and self.last_time is not None:
            dt = (current_time - self.last_time).to_sec()
            if dt > 0.001:
                # Position difference
                dx = pose_msg.position.x - self.last_pose.position.x
                dy = pose_msg.position.y - self.last_pose.position.y
                
                # Get yaw from quaternion (2D approximation)
                qz = pose_msg.orientation.z
                qw = pose_msg.orientation.w
                yaw = 2.0 * np.arctan2(qz, qw)
                
                qz_prev = self.last_pose.orientation.z
                qw_prev = self.last_pose.orientation.w
                yaw_prev = 2.0 * np.arctan2(qz_prev, qw_prev)
                
                # Linear velocity (distance / time)
                dist = np.sqrt(dx**2 + dy**2)
                odom_msg.twist.twist.linear.x = dist / dt
                
                # Angular velocity
                dyaw = yaw - yaw_prev
                # Wrap angle to [-pi, pi]
                while dyaw > np.pi:
                    dyaw -= 2*np.pi
                while dyaw < -np.pi:
                    dyaw += 2*np.pi
                odom_msg.twist.twist.angular.z = dyaw / dt
        
        self.last_pose = pose_msg
        self.last_time = current_time
        
        self._odom_pub.publish(odom_msg)


if __name__ == "__main__":
    gt_pose_visualizer_node = GTPoseVisualizerNode()
    rospy.spin()