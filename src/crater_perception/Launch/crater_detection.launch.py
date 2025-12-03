from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='crater_perception',
            executable='crater_detector',
            name='crater_detector',
            output='screen',
            parameters=[{
                'min_crater_radius': 2.5,
                'max_crater_radius': 15.0,
                'detection_range': 40.0,
                'depth_threshold': 0.3,
            }]
        ),
    ])