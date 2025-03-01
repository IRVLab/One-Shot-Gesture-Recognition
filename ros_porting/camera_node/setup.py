from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'camera_node'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'resource'), glob('resource/*')),
        # ('share/' + package_name, ['resource/g4.json']),
	(os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools', 'opencv-python', 'cv_bridge', 'rosidl_default_runtime'],
    zip_safe=True,
    maintainer='rishikeshjoshi',
    maintainer_email='rjjoshixyz@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'camera_node = camera_node.camera_node:main',
            'pose_detection_node = camera_node.pose_detection_node:main',
            'gesture_recognition_node = camera_node.gesture_recognition_node:main',
            'hand_detection_node = camera_node.hand_detection_node:main',
            'visualization_node = camera_node.visualization_node:main',
        ],
    },
)
