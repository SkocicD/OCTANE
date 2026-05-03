cmake_minimum_required(VERSION 3.8)
project(octane_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/CameraParam.msg"
  "msg/CameraOffset.msg"
  "msg/CameraFrame.msg"
  "msg/DriveCommand.msg"
  "msg/ActuatorCommand.msg"
  "msg/TerrainObject.msg"
  "msg/TerrainObjects.msg"
  "msg/WallSegment.msg"
  "msg/WallSegments.msg"
  DEPENDENCIES sensor_msgs geometry_msgs std_msgs
)

ament_package()
