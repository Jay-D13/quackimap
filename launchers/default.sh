#!/bin/bash

source /environment.sh

source /opt/ros/noetic/setup.bash
source /code/devel/setup.bash --extend

exec roslaunch slam apriltag_ekf_slam_node.launch veh:=$VEHICLE_NAME
