#!/bin/bash
source /environment.sh
source /opt/ros/noetic/setup.bash
source /code/devel/setup.bash --extend

roslaunch encoder_pose encoder_pose_node.launch veh:=$VEHICLE_NAME &

# Launch SLAM node
# roslaunch slam apriltag_ekf_slam_node.launch veh:=$VEHICLE_NAME &
roslaunch map apriltag_map_slam_node.launch veh:=$VEHICLE_NAME &

# Launch ground truth visualizer
# roslaunch gt_pose_visualizer gt_pose_visualizer.launch veh:=$VEHICLE_NAME &

dt-launchfile-join
