#!/bin/bash
# Launch RViz with SLAM visualization configuration

# Replace 'agent' with actual vehicle name in the RViz config
sed -i "s/agent/${VEHICLE_NAME}/g" /opt/ros/noetic/share/rviz/slam_visualization.rviz

# Launch RViz with the SLAM configuration
rviz -d /opt/ros/noetic/share/rviz/slam_visualization.rviz
