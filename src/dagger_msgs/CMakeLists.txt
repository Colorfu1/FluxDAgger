cmake_minimum_required(VERSION 3.0.2)
project(dagger_msgs)

find_package(catkin REQUIRED COMPONENTS
  std_msgs
  sensor_msgs
  message_generation
)

## Declare message files
add_message_files(
  FILES
  ArmStatus.msg
  GlobalState.msg
  InputRequest.msg
  InputResult.msg
  EpisodeInfo.msg
  SyncedObservation.msg
  RewardSignal.msg
)

## Generate added messages
generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
)

## Declare the catkin package
catkin_package(
  CATKIN_DEPENDS std_msgs sensor_msgs message_runtime
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)
