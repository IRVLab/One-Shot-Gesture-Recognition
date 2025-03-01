from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
import os
from datetime import datetime

def generate_launch_description():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bag_output_directory = os.path.join(os.getcwd(), f"rosbag_{timestamp}")
    return LaunchDescription([
        Node(
            package='camera_node',
            executable='pose_detection_node',
            name='pose_detection_node'
        ),
        Node(
            package='camera_node',
            executable='gesture_recognition_node',
            name='gesture_recognition_node',
        ),
        Node(
            package='camera_node',
            executable='hand_detection_node',
            name='hand_detection_node'
        ),
        Node(
            package='camera_node',
            executable='visualization_node',
            name='visualization_node'
        ),
        ExecuteProcess(
            cmd=[
                'ros2', 'bag', 'record',
                '/processed_frames', '/recognized_gesture',
                '--output', bag_output_directory
            ],
        )
    ])
