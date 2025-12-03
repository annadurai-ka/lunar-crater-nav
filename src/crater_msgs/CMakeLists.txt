cmake_minimum_required(VERSION 3.5)
project(crater_msgs)

find_package(ament_cmake REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Register message files
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/CraterInfo.msg"
  DEPENDENCIES builtin_interfaces std_msgs
)

ament_export_dependencies(rosidl_default_runtime)

ament_package()
