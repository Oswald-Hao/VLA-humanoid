cmake_minimum_required(VERSION 3.0.2)
project(vla)

# Find catkin and any other dependencies
find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  nav_msgs
  geometry_msgs
  message_filters
  std_srvs
)

# Declare ROS messages and services
# add_message_files(FILES MyMessage.msg)
# add_service_files(FILES MyService.srv)

# Generate added messages and services
# generate_messages(DEPENDENCIES std_msgs)

# Declare a catkin package
catkin_package()

# Python scripts
catkin_install_python(PROGRAMS
  modules/ros_collector.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Install launch files
install(DIRECTORY
  launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

# Install config files
install(DIRECTORY
  config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
)

# Install other directories
install(DIRECTORY
  docs/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/docs
)

# Install Python modules
install(DIRECTORY
  modules/
  DESTINATION ${CATKIN_PACKAGE_PYTHON_DESTINATION}/vla
  PATTERN "*.py"
  PATTERN "__pycache__" EXCLUDE
)

# Install data directories
install(DIRECTORY
  data/
  trajectories/
  models/
  logs/
  checkpoints/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  PATTERN "*.py" EXCLUDE
)

# Test suite
if(CATKIN_ENABLE_TESTING)
  find_package(rostest REQUIRED)
  # Note: Add rostest files here when available
  # add_rostest_gtest(tests/test_rostest test/test_rostest.test)
endif()

# Export scripts (note: export_runtime_scripts is not a standard cmake command)
# The scripts are already installed via catkin_install_python above