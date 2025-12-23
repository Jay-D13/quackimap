cmake_minimum_required(VERSION 2.8.3)
project(map_slam_scratch)

find_package(catkin REQUIRED COMPONENTS
  duckietown_msgs
  std_msgs
  rospy
  sensor_msgs
  nav_msgs
  geometry_msgs
  visualization_msgs
  cv_bridge
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)
