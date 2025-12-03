cmake_minimum_required(VERSION 3.0.2)
project(ros_vla_vision)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
  geometry_msgs
  message_generation
  kuavo_msgs
  ros_vla_language
  nav_msgs
  trajectory_msgs
  cv_bridge
  image_transport
)

## Find OpenCV
find_package(OpenCV REQUIRED)

## System dependencies
find_package(Boost REQUIRED COMPONENTS system thread)

################################################
## Declare ROS messages, services and actions ##
################################################

## Generate messages in the 'msg' folder
add_message_files(
  FILES
  VLAObjectPose.msg
  VLAObjectDepth.msg
  VLAVLMResult.msg
  VLAGraspPoint.msg
)

## Generate services in the 'srv' folder
add_service_files(
  FILES
  DetectObjects.srv
  EstimatePose.srv
  ExtractFeatures.srv
  GetObjectInfo.srv
)

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
  geometry_msgs
  kuavo_msgs
  ros_vla_language
)

###################################
## catkin specific configuration ##
###################################
catkin_package(
  CATKIN_DEPENDS roscpp rospy std_msgs sensor_msgs geometry_msgs message_runtime 
                 kuavo_msgs ros_vla_language cv_bridge image_transport
  DEPENDS Boost OpenCV
)

###########
## Build ##
###########

## Specify additional locations of header files
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

#############
## Install ##
#############

## Mark executable scripts (Python etc.) for installation
catkin_install_python(PROGRAMS
  scripts/vla_vision_server.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## Mark other files for installation
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
)

install(DIRECTORY scripts/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/scripts
)

install(DIRECTORY models/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/models
)
