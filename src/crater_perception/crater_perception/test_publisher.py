#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
from crater_msgs.msg import CraterInfo
from builtin_interfaces.msg import Time as BITime


class TestPointCloudPublisher(Node):
    def __init__(self):
        super().__init__('test_point_cloud_publisher')
        self.publisher = self.create_publisher(PointCloud2, '/camera/depth/points', 10)
        self.timer = self.create_timer(1.0, self.publish_test_cloud)
        self.crater_info_pub = self.create_publisher(CraterInfo, '/crater_info', 10)
        self.get_logger().info('Publishing test point clouds with synthetic craters')
    
    def publish_test_cloud(self):
        """Generate synthetic point cloud with crater-like features"""
        points = []

        # Generate flat ground
        for x in np.arange(-20, 20, 0.5):
            for y in np.arange(-20, 20, 0.5):
                z = 0.0
                points.append([x, y, z])

        # Add crater 1 at (10, 5) with radius 5m, depth 1m
        crater1_center = np.array([10.0, 5.0])
        crater1_radius = 5.0
        crater1_depth = 1.0

        for x in np.arange(5, 15, 0.3):
            for y in np.arange(0, 10, 0.3):
                dist = np.linalg.norm([x - crater1_center[0], y - crater1_center[1]])
                if dist < crater1_radius:
                    # Parabolic crater profile
                    z = -crater1_depth * (1 - (dist / crater1_radius)**2)
                    points.append([x, y, z])

        # Add crater 2 at (-5, -8) with radius 7m, depth 1.5m
        crater2_center = np.array([-5.0, -8.0])
        crater2_radius = 7.0
        crater2_depth = 1.5

        for x in np.arange(-12, 2, 0.3):
            for y in np.arange(-15, -1, 0.3):
                dist = np.linalg.norm([x - crater2_center[0], y - crater2_center[1]])
                if dist < crater2_radius:
                    z = -crater2_depth * (1 - (dist / crater2_radius)**2)
                    points.append([x, y, z])

        # Create PointCloud2 message with proper header
        header = Header()
        now_msg = self.get_clock().now().to_msg()
        header.stamp = now_msg
        header.frame_id = 'base_link'

        cloud_msg = pc2.create_cloud_xyz32(header, points)
        self.publisher.publish(cloud_msg)

        # Also publish CraterInfo messages for the two synthetic craters
        # Build CraterInfo objects and reuse same timestamp
        ci1 = CraterInfo()
        ci1.x = float(crater1_center[0])
        ci1.y = float(crater1_center[1])
        ci1.radius = float(crater1_radius)
        ci1.confidence = 0.95
        ci1.inlier_ratio = 0.9
        ci1.stamp = BITime(sec=now_msg.sec, nanosec=now_msg.nanosec)
        ci1.id_guess = ''
        self.crater_info_pub.publish(ci1)

        ci2 = CraterInfo()
        ci2.x = float(crater2_center[0])
        ci2.y = float(crater2_center[1])
        ci2.radius = float(crater2_radius)
        ci2.confidence = 0.95
        ci2.inlier_ratio = 0.9
        ci2.stamp = BITime(sec=now_msg.sec, nanosec=now_msg.nanosec)
        ci2.id_guess = ''
        self.crater_info_pub.publish(ci2)

        self.get_logger().info(f'Published cloud with {len(points)} points, expecting 2 craters')

def main(args=None):
    rclpy.init(args=args)
    node = TestPointCloudPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()