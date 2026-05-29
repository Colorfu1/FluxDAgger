cmake_minimum_required(VERSION 3.0.2)
project(dagger)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  dagger_msgs
)

# Register src/dagger/*.py and subpackages as the Python package `dagger`
# so node entry scripts can consistently use `from dagger.xxx import yyy`.
catkin_python_setup()

catkin_package(
  CATKIN_DEPENDS rospy std_msgs sensor_msgs dagger_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(
  PROGRAMS
    scripts/dagger_controller_node.py
    scripts/dagger_collector_node.py
    scripts/arm_node.py
    scripts/sync_observation_node.py
    scripts/qwen3_reward_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
  PATTERN ".svn" EXCLUDE
)

install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
  PATTERN ".svn" EXCLUDE
)
