#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseArray, Pose
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
import cv2
from crater_msgs.msg import CraterInfo
from builtin_interfaces.msg import Time as BITime

class CraterDetector(Node):
    def __init__(self):
        super().__init__('crater_detector')
        
        # Parameters
        self.declare_parameter('min_crater_radius', 2.5)
        self.declare_parameter('max_crater_radius', 15.0)
        self.declare_parameter('detection_range', 40.0)
        self.declare_parameter('depth_threshold', 0.2)
        
        self.min_radius = self.get_parameter('min_crater_radius').value
        self.max_radius = self.get_parameter('max_crater_radius').value
        self.detection_range = self.get_parameter('detection_range').value
        self.depth_thresh = self.get_parameter('depth_threshold').value
        
        # Subscribers
        self.depth_sub = self.create_subscription(
            PointCloud2,
            '/camera/depth/points',
            self.point_cloud_callback,
            10
        )
        
        # Publishers
        self.crater_pub = self.create_publisher(
            PoseArray,
            '/crater_detections',
            10
        )
        self.crater_info_pub = self.create_publisher(CraterInfo, '/crater_info', 10)
        
        self.get_logger().info('=== HOUGH CIRCLE CRATER DETECTOR V2 ===')
        self.get_logger().info(f'Min radius: {self.min_radius}m, Max radius: {self.max_radius}m')
    
    def point_cloud_callback(self, msg):
        try:
            points = self.pointcloud2_to_array(msg)
            
            if len(points) < 100:
                return
            
            craters = self.detect_craters_hough(points)
            self.publish_craters(craters, msg.header)
                
        except Exception as e:
            self.get_logger().error(f'Error: {str(e)}')
    
    def pointcloud2_to_array(self, cloud_msg):
        points_list = []
        for point in pc2.read_points(cloud_msg, skip_nans=True, 
                                     field_names=("x", "y", "z")):
            distance = np.sqrt(point[0]**2 + point[1]**2)
            if distance < self.detection_range:
                points_list.append([point[0], point[1], point[2]])
        return np.array(points_list)
    
    def detect_craters_hough(self, points):
        craters = []
        
        if len(points) == 0:
            return []
        
        # Create grid
        grid_res = 0.3
        x_min, x_max = points[:, 0].min(), points[:, 0].max()
        y_min, y_max = points[:, 1].min(), points[:, 1].max()
        
        x_bins = int((x_max - x_min) / grid_res) + 1
        y_bins = int((y_max - y_min) / grid_res) + 1
        
        if x_bins < 10 or y_bins < 10:
            return []
        
        self.get_logger().info(f'Grid: {x_bins}x{y_bins}, {len(points)} points')
        
        # Create depth image
        depth_image = np.full((y_bins, x_bins), np.inf)
        
        for point in points:
            x_idx = int((point[0] - x_min) / grid_res)
            y_idx = int((point[1] - y_min) / grid_res)
            if 0 <= x_idx < x_bins and 0 <= y_idx < y_bins:
                depth_image[y_idx, x_idx] = min(depth_image[y_idx, x_idx], point[2])
        
        valid_depths = depth_image[depth_image != np.inf]
        if len(valid_depths) == 0:
            return []
        
        median_depth = np.median(valid_depths)
        depth_image[depth_image == np.inf] = median_depth
        
        self.get_logger().info(f'Depth: min={valid_depths.min():.2f}, max={valid_depths.max():.2f}')
        
        # Normalize and invert
        depth_normalized = ((depth_image - depth_image.min()) / 
                           (depth_image.max() - depth_image.min() + 1e-6) * 255).astype(np.uint8)
        depth_inverted = 255 - depth_normalized
        
        # Threshold
        threshold_value = 50  # Fixed threshold for testing
        _, binary = cv2.threshold(depth_inverted, threshold_value, 255, cv2.THRESH_BINARY)
        
        # Clean up
        kernel = np.ones((3,3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Detect circles
        circles = cv2.HoughCircles(
            binary,
            cv2.HOUGH_GRADIENT,
            dp=1.5,
            minDist=int(self.min_radius * 2 / grid_res),
            param1=50,
            param2=15,
            minRadius=int(self.min_radius / grid_res),
            maxRadius=int(self.max_radius / grid_res)
        )
        
        if circles is None:
            self.get_logger().info('No circles found')
            return []
        
        circles = np.uint16(np.around(circles))
        self.get_logger().info(f'*** FOUND {len(circles[0])} CIRCLES ***')
        
        # Convert to world coordinates
        for circle in circles[0, :]:
            x_img, y_img, r_img = circle
            
            if x_img >= x_bins or y_img >= y_bins:
                continue
            
            x_world = x_min + x_img * grid_res
            y_world = y_min + y_img * grid_res
            r_world = r_img * grid_res
            
            center_depth = depth_image[min(y_img, y_bins-1), min(x_img, x_bins-1)]
            crater_depth = median_depth - center_depth
            
            if self.min_radius <= r_world <= self.max_radius:
                craters.append({
                    'x': float(x_world),
                    'y': float(y_world),
                    'radius': float(r_world),
                    'confidence': 0.8
                })
                self.get_logger().info(f'  -> Crater at ({x_world:.1f}, {y_world:.1f}), r={r_world:.1f}m')
        
        return craters
    
    def publish_craters(self, craters, header):
        # PoseArray (existing)
        pose_array = PoseArray()
        pose_array.header = header
        pose_array.header.frame_id = 'base_link'
        for crater in craters:
            pose = Pose()
            pose.position.x = crater['x']
            pose.position.y = crater['y']
            pose.position.z = crater['radius']
            pose.orientation.w = crater.get('confidence', 1.0)
            pose_array.poses.append(pose)
        self.crater_pub.publish(pose_array)

        # Publish individual CraterInfo messages for each detection
        for crater in craters:
            ci = CraterInfo()
            ci.x = float(crater['x'])
            ci.y = float(crater['y'])
            ci.radius = float(crater['radius'])
            ci.confidence = float(crater.get('confidence', 1.0))
            ci.inlier_ratio = float(crater.get('inlier_ratio', 0.0))
            # header stamp -> builtin_interfaces/Time
            ci.stamp = BITime(sec=header.stamp.sec, nanosec=header.stamp.nanosec)
            ci.id_guess = crater.get('id_guess', '')
            self.crater_info_pub.publish(ci)



def main(args=None):
    rclpy.init(args=args)
    node = CraterDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()