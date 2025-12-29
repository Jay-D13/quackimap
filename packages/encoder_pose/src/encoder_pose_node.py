#!/usr/bin/env python3
import os
import time
from typing import Optional

import numpy as np
import rospy
import yaml
from duckietown.dtros import DTROS, NodeType, TopicType
from duckietown_msgs.msg import Twist2DStamped, WheelEncoderStamped, EpisodeStart
from nav_msgs.msg import Odometry

from encoder_pose.include.odometry.odometry import delta_phi, estimate_pose

from multiprocessing import Lock

class EncoderPoseNode(DTROS):
    """
    Computes an estimate of the Duckiebot pose using the wheel encoders.
    Args:
        node_name (:obj:`str`): a unique, descriptive name for the ROS node
    Configuration:

    Publisher:
        ~encoder_localization (:obj:`PoseStamped`): The computed position
    Subscribers:
        ~/left_wheel_encoder_node/tick (:obj:`WheelEncoderStamped`):
            encoder ticks
        ~/right_wheel_encoder_node/tick (:obj:`WheelEncoderStamped`):
            encoder ticks
    """

    right_tick_prev: Optional[int]
    left_tick_prev: Optional[int]
    delta_phi_left: float
    delta_phi_right: float

    def __init__(self, node_name):
        # Initialize the DTROS parent class
        super(EncoderPoseNode, self).__init__(node_name=node_name, node_type=NodeType.LOCALIZATION)
        self.log("Initializing...")
        # get the name of the robot
        self.veh = rospy.get_namespace().strip("/")

        self.right_wheel_mutex = Lock()
        self.left_wheel_mutex = Lock()

        # Init the parameters
        self.resetParameters()

        self.odom_topic = rospy.get_param("~odom_topic", "encoder_pose_node/odom")

        # nominal R and L, you may change these if needed:

        self.R = 0.0318  # meters, default value of wheel radius
        self.baseline = 0.11  # meters, default value of baseline for DB21

        # Defining subscribers:

        # Wheel encoder subscriber:
        left_encoder_topic = f"/{self.veh}/left_wheel_encoder_driver_node/tick"
        rospy.Subscriber(left_encoder_topic, WheelEncoderStamped, self.cbLeftEncoder)

        # Wheel encoder subscriber:
        right_encoder_topic = f"/{self.veh}/right_wheel_encoder_driver_node/tick"
        rospy.Subscriber(right_encoder_topic, WheelEncoderStamped, self.cbRightEncoder)

        
        # Odometry publisher (using relative name so launch file remap works)
        self.db_estimated_pose = rospy.Publisher(
            self.odom_topic, Odometry, queue_size=1, dt_topic_type=TopicType.LOCALIZATION
        )

        self.log("Initialized.")
        rospy.Timer(rospy.Duration(0.05), self.posePublisher)

    def resetParameters(self):
        # Add the node parameters to the parameters dictionary
        self.delta_phi_left = 0.0
        self.left_tick_prev = None

        self.delta_phi_right = 0.0
        self.right_tick_prev = None

        # Initializing the odometry
        self.x_prev = 0.0
        self.y_prev = 0.0
        self.theta_prev = 0.0

        # Timestamp for velocity calculation
        self.last_update_time = None


    def cbLeftEncoder(self, encoder_msg):
        """
        Wheel encoder callback
        Args:
            encoder_msg (:obj:`WheelEncoderStamped`) encoder ROS message.
        """
        with self.left_wheel_mutex:
            # initializing ticks to stored absolute value
            if self.left_tick_prev is None:
                self.left_tick_prev = encoder_msg.data
                return

            left_ticks_curr = encoder_msg.data

            # running the DeltaPhi() function copied from the notebooks to calculate rotations
            delta_phi_left = delta_phi(
                left_ticks_curr, self.left_tick_prev, encoder_msg.resolution
            )
            if delta_phi_left == 0:
                return
            self.left_tick_prev = left_ticks_curr
            self.delta_phi_left += delta_phi_left

    def cbRightEncoder(self, encoder_msg):
        """
        Wheel encoder callback, the rotation of the wheel.
        Args:
            encoder_msg (:obj:`WheelEncoderStamped`) encoder ROS message.
        """

        with self.right_wheel_mutex:
            if self.right_tick_prev is None:
                self.right_tick_prev = encoder_msg.data
                return

            right_ticks_curr = encoder_msg.data

            # calculate rotation of right wheel
            delta_phi_right = delta_phi(
                right_ticks_curr, self.right_tick_prev, encoder_msg.resolution
            )
            if delta_phi_right == 0:
                return
            self.right_tick_prev = right_ticks_curr
            self.delta_phi_right += delta_phi_right

    def posePublisher(self, event=None):
        """
        Publish the pose of the Duckiebot given by the kinematic model
            using the encoders.
        Publish:
            ~/pose (:obj:`PoseStamped`): Duckiebot pose.
        """
        with self.left_wheel_mutex:
            with self.right_wheel_mutex:

                x_curr, y_curr, theta_curr = estimate_pose(
                    self.R,
                    self.baseline,
                    self.x_prev,
                    self.y_prev,
                    self.theta_prev,
                    self.delta_phi_left,
                    self.delta_phi_right,
                )
                """ 
                if (x_curr == self.x_prev) and (y_curr == self.y_prev) and (theta_curr == self.theta_prev):
                    return
                """
                theta_curr = self.angle_clamp(theta_curr)  # angle always between 0,2pi

                # Calculate dt and velocities
                current_time = rospy.Time.now()
                if self.last_update_time is None:
                    self.last_update_time = current_time
                    # Skip first iteration, can't compute velocity without dt
                    self.delta_phi_left = 0
                    self.delta_phi_right = 0
                    self.x_prev = x_curr
                    self.y_prev = y_curr
                    self.theta_prev = theta_curr
                    return

                dt = (current_time - self.last_update_time).to_sec()
                if dt <= 0.0:
                    return

                # Compute wheel velocities
                d_left = self.R * self.delta_phi_left  # distance traveled by left wheel
                d_right = self.R * self.delta_phi_right  # distance traveled by right wheel
                d_center = (d_left + d_right) / 2.0  # distance traveled by robot center

                # Linear and angular velocities
                v = d_center / dt  # linear velocity (m/s)
                w = (d_right - d_left) / (self.baseline * dt)  # angular velocity (rad/s)
                """                 
                # self.logging to screen for debugging purposes
                self.log("              ODOMETRY             ")
                # self.log(f"Baseline : {self.baseline}   R: {self.R}")
                self.log("Just this move:")
                self.log(f"Theta : {np.rad2deg(theta_curr) - np.rad2deg(self.theta_prev)} deg,  x: {x_curr - self.x_prev} m,  y: {y_curr - self.y_prev} m")
                self.log("Total accumulated:")
                self.log(f"Theta : {np.rad2deg(theta_curr)} deg,  x: {x_curr} m,  y: {y_curr} m")
                self.log(f"Velocities: v={v:.3f} m/s, w={np.rad2deg(w):.1f} deg/s")

                self.log(
                    f"Rotation left wheel : {np.rad2deg(self.delta_phi_left)} deg,   "
                    f"Rotation right wheel : {np.rad2deg(self.delta_phi_right)} deg"
                )
                self.log(f"Prev Ticks left : {self.left_tick_prev}   Prev Ticks right : {self.right_tick_prev}")
                # self.log(
                #     f"Prev integral error : {self.prev_int}")
                """
                # Calculate new odometry only when new data from encoders arrives
                self.delta_phi_left = 0
                self.delta_phi_right = 0

                # Current estimate becomes previous estimate at next iteration
                self.x_prev = x_curr
                self.y_prev = y_curr
                self.theta_prev = theta_curr
                self.last_update_time = current_time

                # Creating message to plot pose in RVIZ
                odom = Odometry()
                odom.header.frame_id = "map"
                odom.header.stamp = current_time

                odom.pose.pose.position.x = x_curr  # x position - estimate
                odom.pose.pose.position.y = y_curr  # y position - estimate
                odom.pose.pose.position.z = 0  # z position - no flying allowed in Duckietown

                # these are quaternions!
                odom.pose.pose.orientation.x = 0
                odom.pose.pose.orientation.y = 0
                odom.pose.pose.orientation.z = np.sin(theta_curr / 2)
                odom.pose.pose.orientation.w = np.cos(theta_curr / 2)

                # added velocity information for SLAM prediction
                odom.twist.twist.linear.x = v
                odom.twist.twist.linear.y = 0.0
                odom.twist.twist.linear.z = 0.0
                odom.twist.twist.angular.x = 0.0
                odom.twist.twist.angular.y = 0.0
                odom.twist.twist.angular.z = w

                self.db_estimated_pose.publish(odom)



    def onShutdown(self):
        super(EncoderPoseNode, self).on_shutdown()

    @staticmethod
    def angle_clamp(theta):
        if theta > 2 * np.pi:
            return theta - 2 * np.pi
        elif theta < -2 * np.pi:
            return theta + 2 * np.pi
        else:
            return theta


if __name__ == "__main__":
    # Initialize the node
    encoder_pose_node = EncoderPoseNode(node_name="encoder_pose_node")
    # Keep it spinning
    rospy.spin()