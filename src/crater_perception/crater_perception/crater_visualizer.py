#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray
from visualization_msgs.msg import Marker, MarkerArray

class CraterVisualizer(Node):
    def __init__(self):
        super().__init__('crater_visualizer')
        
        # Subscribe to crater detections
        self.detection_sub = self.create_subscription(
            PoseArray,
            '/crater_detections',
            self.visualize_craters,
            10
        )
        
        # Publish visualization markers
        self.marker_pub = self.create_publisher(
            MarkerArray,
            '/crater_markers',
            10
        )
        
        self.get_logger().info('Crater Visualizer Started')
    
    def visualize_craters(self, msg):
        """Convert crater detections to visualization markers"""
        marker_array = MarkerArray()
        
        for i, pose in enumerate(msg.poses):
            # Extract crater info
            crater_x = pose.position.x
            crater_y = pose.position.y
            crater_radius = pose.position.z
            confidence = pose.orientation.w
            
            # Create sphere marker for crater location
            marker = Marker()
            marker.header = msg.header
            marker.header.frame_id = 'base_link'
            marker.ns = 'craters'
            marker.id = i * 2  # Even IDs for spheres
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            
            # Position at crater center
            marker.pose.position.x = crater_x
            marker.pose.position.y = crater_y
            marker.pose.position.z = -0.5  # Slightly below ground
            marker.pose.orientation.w = 1.0
            
            # Size based on crater radius
            marker.scale.x = crater_radius * 2  # Diameter
            marker.scale.y = crater_radius * 2
            marker.scale.z = 0.5  # Flat sphere
            
            # Color: green for high confidence, yellow for low
            marker.color.r = 1.0 - confidence
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 0.5  # Semi-transparent
            
            marker.lifetime.sec = 1
            marker_array.markers.append(marker)
            
            # Create text label
            text_marker = Marker()
            text_marker.header = msg.header
            text_marker.header.frame_id = 'base_link'
            text_marker.ns = 'crater_labels'
            text_marker.id = i * 2 + 1  # Odd IDs for labels
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            
            text_marker.pose.position.x = crater_x
            text_marker.pose.position.y = crater_y
            text_marker.pose.position.z = 1.0  # Above ground
            text_marker.pose.orientation.w = 1.0
            
            text_marker.text = f'R={crater_radius:.1f}m\nC={confidence:.2f}'
            text_marker.scale.z = 0.5  # Text size
            
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.color.a = 1.0
            
            text_marker.lifetime.sec = 1
            marker_array.markers.append(text_marker)
        
        self.marker_pub.publish(marker_array)
        self.get_logger().info(f'Published {len(msg.poses)} crater markers', throttle_duration_sec=2.0)

def main(args=None):
    rclpy.init(args=args)
    node = CraterVisualizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()