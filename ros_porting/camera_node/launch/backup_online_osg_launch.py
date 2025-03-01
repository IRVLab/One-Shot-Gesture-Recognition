from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='camera_node',
            executable='pose_detection_node',
            name='pose_detection_node'
        ),
        Node(
            package='camera_node',
            executable='hand_detection_node',
            name='hand_detection_node'
        ),
        Node(
            package='camera_node',
            executable='gesture_recognition_node',
            name='gesture_recognition_node',
                
        )
    ])
