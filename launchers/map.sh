#!/bin/bash
source /environment.sh
source /opt/ros/noetic/setup.bash
source /code/devel/setup.bash --extend

roslaunch encoder_pose encoder_pose_node.launch veh:=$VEHICLE_NAME &

# Launch SLAM node
roslaunch map_slam apriltag_map_slam_node.launch veh:=$VEHICLE_NAME &

dt-launchfile-join
