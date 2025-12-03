#!/bin/bash
source /environment.sh
source /opt/ros/noetic/setup.bash
source /code/devel/setup.bash --extend

roslaunch slam apriltag_ekf_slam_node.launch veh:=$VEHICLE_NAME &
roslaunch gt_pose_visualizer gt_pose_visualizer_node.launch veh:=$VEHICLE_NAME &

dt-launchfile-join