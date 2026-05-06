cmake_minimum_required(VERSION 3.8)
project(octane_terrain_mapping)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/TerrainObject.msg"
  "msg/TerrainObjects.msg"
  "msg/WallSegment.msg"
  "msg/WallSegments.msg"
  DEPENDENCIES std_msgs
)

ament_package()
