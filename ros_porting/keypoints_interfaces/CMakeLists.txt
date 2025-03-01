cmake_minimum_required(VERSION 3.5)
project(keypoints_interfaces)

# Find dependencies
find_package(rosidl_default_generators REQUIRED)

# Generate messages
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/PoseKeypoints.msg"
  "msg/PoseKeypointsArray.msg"
  "msg/HandKeypoints.msg"
  "msg/HandKeypointsArray.msg"
)

# Export the package
ament_package()
