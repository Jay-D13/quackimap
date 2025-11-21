#!/usr/bin/env python3

import asyncio

import rospy
from geometry_msgs.msg import Pose
from dt_robot_utils import get_robot_name
from duckietown.sdk.robots.duckiebot import DB21J
from dtps import context, ContextConfig
from dtps_http import RawData
from duckietown.dtros import DTROS, NodeType, TopicType
from duckietown_messages.utils.exceptions import DataDecodingError



class GTPoseVisualizerNode(DTROS):

    def __init__(self):
        super(GTPoseVisualizerNode, self).__init__(node_name="gt_pose_visualizer_node", node_type=NodeType.DRIVER)
        self._robot_name = get_robot_name()
        # arguments

        # Here we use the Duckietown SDK to connect directly to the entity in the duckiematrix to get the
        # robot pose
        # TODO would be better to load the robot name from the settings.yaml file
        self.robot: DB21J = DB21J("map/vehicle_0", simulated=True)
        self.robot.pose.start()

        # create publisher
        self._pub = rospy.Publisher(
            "~gt_pose",
            Pose,
            queue_size=1,
            dt_topic_type=TopicType.DRIVER,
            dt_help="The ground truth pose of the robot from the Duckiematrix",
        )

        rospy.Timer(rospy.Duration(0.1), self.publish_pose)


    def publish_pose(self):
        pose = self.robot.pose.capture()
        pose_msg = Pose()
        pose_msg.position.x = pose["position"]["x"]
        pose_msg.position.y = pose["position"]["y"]
        pose_msg.position.z = pose["position"]["z"]
        pose_msg.orientation.x = pose["rotation"]["x"]
        pose_msg.orientation.y = pose["rotation"]["y"]
        pose_msg.orientation.z = pose["rotation"]["z"]
        pose_msg.orientation.w = pose["rotation"]["w"]
        self._pub.publish(pose_msg)

if __name__ == "__main__":
    gt_pose_visualizer_node = GTPoseVisualizerNode()
    rospy.spin()
