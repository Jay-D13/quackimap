#!/bin/bash
set -e

source /environment.sh

# Initialize launchfile (Duckietown helper)
dt-launchfile-init

rosrun slam apriltag_ekf_slam_node.py \
    _image_topic:=camera_node/image/compressed \
    _odom_topic:=velocity_to_pose_node/pose \
    _camera_info_topic:=camera_node/camera_info \
    _tag_size:=0.065

# Wait for node to exit
dt-launchfile-join
