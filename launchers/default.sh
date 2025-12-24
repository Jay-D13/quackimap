#!/bin/bash
source /environment.sh
source /opt/ros/noetic/setup.bash
source /code/devel/setup.bash --extend

roslaunch encoder_pose encoder_pose_node.launch veh:=$VEHICLE_NAME &

# Launch SLAM node
# roslaunch ekf_slam apriltag_ekf_slam_node.launch veh:=$VEHICLE_NAME &
roslaunch gtsam_slam apriltag_gtsam_slam_node.launch veh:=$VEHICLE_NAME &

dt-launchfile-join
