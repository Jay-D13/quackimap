#!/bin/bash

source /environment.sh

dt-launchfile-init


source /opt/ros/noetic/setup.bash
source /code/devel/setup.bash --extend

exec roslaunch ekf_localization ekf_localization_node.launch veh:=$VEHICLE_NAME
exec roslaunch gt_pose_visualizer gt_pose_visualizer_node.launch

dt-launchfile-join