#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Image
from geometry_msgs.msg import PoseArray, Pose
from cv_bridge import CvBridge
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
import cv2

class CraterDetector(Node):
    def __init__(self):
        super().__init__('crater_detector')
        
        # Parameters
        self.declare_parameter('min_crater_radius', 2.5)  # meters
        self.declare_parameter('max_crater_radius', 15.0)  # meters
        self.declare_parameter('detection_range', 40.0)  # meters
        self.declare_parameter('depth_threshold', 0.2)  # meters (crater depth)
        
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
        
        self.get_logger().info('Crater Detector Node Initialized (Hough Circle Method)')
        self.get_logger().info(f'Detection params: radius {self.min_radius}-{self.max_radius}m, range {self.detection_range}m')
    
    def point_cloud_callback(self, msg):
        """Main crater detection pipeline"""
        try:
            # Convert ROS PointCloud2 to numpy array
            points = self.pointcloud2_to_array(msg)
            
            if len(points) < 100:
                self.get_logger().warn('Insufficient points in cloud')
                return
            
            # Detect craters using Hough circles on depth image
            craters = self.detect_craters_hough(points)
            
            # Publish detections
            self.publish_craters(craters, msg.header)
                
        except Exception as e:
            self.get_logger().error(f'Error in crater detection: {str(e)}')
            import traceback
            self.get_logger().error(traceback.format_exc())
    
    def pointcloud2_to_array(self, cloud_msg):
        """Convert ROS PointCloud2 to numpy array"""
        points_list = []
        
        for point in pc2.read_points(cloud_msg, skip_nans=True, 
                                     field_names=("x", "y", "z")):
            # Filter points within detection range
            distance = np.sqrt(point[0]**2 + point[1]**2)
            if distance < self.detection_range:
                points_list.append([point[0], point[1], point[2]])
        
        return np.array(points_list)
    
    def detect_craters_hough(self, points):
        """
        Detect craters using Hough Circle Transform on 2D projection
        This is more robust for sparse point clouds
        """
        craters = []
        
        if len(points) == 0:
            return []
        
        # 1. Create occupancy grid and depth map
        grid_res = 0.3  # meters per pixel (higher resolution for better detection)
        
        x_min, x_max = points[:, 0].min(), points[:, 0].max()
        y_min, y_max = points[:, 1].min(), points[:, 1].max()
        
        x_bins = int((x_max - x_min) / grid_res) + 1
        y_bins = int((y_max - y_min) / grid_res) + 1
        
        if x_bins < 10 or y_bins < 10:
            self.get_logger().warn(f'Grid too small: {x_bins}x{y_bins}')
            return []
        
        self.get_logger().info(f'Processing {len(points)} points into {x_bins}x{y_bins} grid')
        
        # Create depth image
        depth_image = np.full((y_bins, x_bins), np.inf)
        
        for point in points:
            x_idx = int((point[0] - x_min) / grid_res)
            y_idx = int((point[1] - y_min) / grid_res)
            
            if 0 <= x_idx < x_bins and 0 <= y_idx < y_bins:
                depth_image[y_idx, x_idx] = min(depth_image[y_idx, x_idx], point[2])
        
        # Fill infinite values with median
        valid_depths = depth_image[depth_image != np.inf]
        if len(valid_depths) == 0:
            return []
        
        median_depth = np.median(valid_depths)
        depth_image[depth_image == np.inf] = median_depth
        
        self.get_logger().info(f'Depth range: [{valid_depths.min():.2f}, {valid_depths.max():.2f}], median: {median_depth:.2f}')
        
        # 2. Create binary image of depressions
        # Normalize depth to 0-255 range
        depth_normalized = ((depth_image - depth_image.min()) / 
                           (depth_image.max() - depth_image.min() + 1e-6) * 255).astype(np.uint8)
        
        # Invert so depressions are bright
        depth_inverted = 255 - depth_normalized
        
        # Threshold to find depressions
        threshold_value = int(255 * (self.depth_thresh / (depth_image.max() - depth_image.min() + 0.01)))
        _, binary = cv2.threshold(depth_inverted, threshold_value, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations to clean up
        kernel = np.ones((3,3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # 3. Detect circles using Hough transform
        circles = cv2.HoughCircles(
            binary,
            cv2.HOUGH_GRADIENT,
            dp=1.5,
            minDist=int(self.min_radius * 2 / grid_res),
            param1=50,
            param2=15,  # Lower threshold for easier detection
            minRadius=int(self.min_radius / grid_res),
            maxRadius=int(self.max_radius / grid_res)
        )
        
        if circles is None:
            self.get_logger().info('No circles detected by Hough transform')
            return []
        
        circles = np.uint16(np.around(circles))
        self.get_logger().info(f'Hough detected {len(circles[0])} circles')
        
        # 4. Convert circles to world coordinates
        for circle in circles[0, :]:
            x_img, y_img, r_img = circle
            
            # Check bounds
            if x_img >= x_bins or y_img >= y_bins:
                continue
            
            # Convert to world coordinates
            x_world = x_min + x_img * grid_res
            y_world = y_min + y_img * grid_res
            r_world = r_img * grid_res
            
            # Get average depth at crater center
            center_depth = depth_image[min(y_img, y_bins-1), min(x_img, x_bins-1)]
            crater_depth = median_depth - center_depth
            
            # Filter valid craters
            if (self.min_radius <= r_world <= self.max_radius and 
                crater_depth > 0):
                
                confidence = min(1.0, crater_depth / self.depth_thresh)
                
                craters.append({
                    'x': float(x_world),
                    'y': float(y_world),
                    'radius': float(r_world),
                    'depth': float(crater_depth),
                    'confidence': float(confidence)
                })
                
                self.get_logger().info(
                    f'Crater: pos=({x_world:.1f}, {y_world:.1f}), '
                    f'radius={r_world:.1f}m, depth={crater_depth:.2f}m'
                )
        
        self.get_logger().info(f'Final count: {len(craters)} valid craters')
        return craters
    
    def publish_craters(self, craters, header):
        """Publish detected craters as PoseArray"""
        pose_array = PoseArray()
        pose_array.header = header
        pose_array.header.frame_id = 'base_link'
        
        for crater in craters:
            pose = Pose()
            pose.position.x = crater['x']
            pose.position.y = crater['y']
            pose.position.z = crater['radius']  # Store radius in z
            pose.orientation.w = crater.get('confidence', 1.0)  # Store confidence in w
            
            pose_array.poses.append(pose)
        
        self.crater_pub.publish(pose_array)
        self.get_logger().info(f'Published {len(craters)} crater detections')

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