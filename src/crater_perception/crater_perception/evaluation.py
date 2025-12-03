#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray
import numpy as np

class CraterEvaluator(Node):
    def __init__(self):
        super().__init__('crater_evaluator')
        
        # Ground truth craters (x, y, radius)
        self.ground_truth = [
            {'x': 10.0, 'y': 5.0, 'radius': 5.0},
            {'x': -5.0, 'y': -8.0, 'radius': 7.0},
        ]
        
        self.detection_sub = self.create_subscription(
            PoseArray,
            '/crater_detections',
            self.evaluate_detections,
            10
        )
        
        self.get_logger().info('Crater Evaluator Started')
    
    def evaluate_detections(self, msg):
        """Compare detections to ground truth"""
        detected = []
        for pose in msg.poses:
            detected.append({
                'x': pose.position.x,
                'y': pose.position.y,
                'radius': pose.position.z
            })
        
        # Calculate metrics
        tp, fp, fn = self.calculate_metrics(detected, self.ground_truth)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        self.get_logger().info(f'Detection Metrics:')
        self.get_logger().info(f'  True Positives: {tp}')
        self.get_logger().info(f'  False Positives: {fp}')
        self.get_logger().info(f'  False Negatives: {fn}')
        self.get_logger().info(f'  Precision: {precision:.2f}')
        self.get_logger().info(f'  Recall: {recall:.2f}')
    
    def calculate_metrics(self, detected, ground_truth, distance_thresh=3.0):
        """Calculate TP, FP, FN"""
        matched_gt = set()
        tp = 0
        
        for det in detected:
            matched = False
            for i, gt in enumerate(ground_truth):
                if i in matched_gt:
                    continue
                
                dist = np.sqrt((det['x'] - gt['x'])**2 + (det['y'] - gt['y'])**2)
                
                if dist < distance_thresh:
                    tp += 1
                    matched_gt.add(i)
                    matched = True
                    break
        
        fp = len(detected) - tp
        fn = len(ground_truth) - len(matched_gt)
        
        return tp, fp, fn

def main(args=None):
    rclpy.init(args=args)
    node = CraterEvaluator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()