#!/usr/bin/env python3
# ============================================================
# ros2_nodes/robot_classifier_node.py
# Week 7 CamVis subscriber pattern.
#
# Subscribes : /depth_cam/rgb/image_raw
# Publishes  : /object_classifier/prediction (std_msgs/String JSON)
#
# Detects ROIs, classifies ALL loaded classes, publishes best.
#
# Usage (after colcon build + source install/setup.zsh):
#   ros2 run robot_classifier robot_classifier \
#     --model-path /path/to/robot_finetuned_model.pth \
#     --class-map  /path/to/robot_finetuned_class_to_idx.json
# ============================================================

import sys, os, json, argparse
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
import cv2
from cv_bridge import CvBridge
import torch

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_DIR, '..'))
sys.path.insert(0, _DIR)

from utils.model_utils     import load_checkpoint, get_device
from utils.inference_utils import (get_inference_transform,
                                   detect_and_classify, draw_detections)


class RobotClassifierNode(Node):

    def __init__(self, model_path, class_map_path, conf_threshold=0.70):
        super().__init__('robot_classifier')

        # Camera subscriber — Week 7/8 pattern
        self.cam_subscription = self.create_subscription(
            Image, '/depth_cam/rgb/image_raw',
            self.image_callback, 1)

        # Prediction publisher
        self.pred_publisher = self.create_publisher(
            String, '/object_classifier/prediction', 10)

        # cv_bridge — Week 7/8 pattern
        self.cv_bridge = CvBridge()

        # Load model — all classes
        self.device = get_device()
        self.model, self.class_to_idx, self.idx_to_class = \
            load_checkpoint(model_path, device=str(self.device))
        self.model.eval()

        self.transform      = get_inference_transform(224)
        self.conf_threshold = conf_threshold
        self.roi_params = {
            'blur_kernel':7, 'canny_low':30, 'canny_high':100,
            'dilate_iters':3, 'min_area':4000,
            'max_area_ratio':0.85, 'padding':10,
        }

        self.get_logger().info(
            f'RobotClassifierNode ready | '
            f'{len(self.class_to_idx)} classes | '
            f'threshold={self.conf_threshold}')

    def image_callback(self, msg):
        # Week 7 pattern
        image_bgr = self.cv_bridge.imgmsg_to_cv2(msg, 'bgr8')

        detections = detect_and_classify(
            image_bgr, self.model, self.idx_to_class,
            self.transform, self.device,
            self.conf_threshold, **self.roi_params)

        annotated = draw_detections(image_bgr, detections, self.conf_threshold)
        cv2.imshow('robot_classifier', annotated)
        cv2.waitKey(1)

        above = [d for d in detections if d['above_threshold']]
        if above:
            best = max(above, key=lambda d: d['confidence'])
            payload = {
                'label':          best['label'],
                'confidence':     round(best['confidence'], 4),
                'bbox':           list(best['bbox']),
                'num_detections': len(above),
                'all_detections': [
                    {'label': d['label'],
                     'confidence': round(d['confidence'], 4),
                     'bbox': list(d['bbox'])} for d in above],
            }
            self.get_logger().info(
                f"Best: {payload['label']} conf={payload['confidence']:.3f}")
        else:
            payload = {'label':'','confidence':0.0,
                       'bbox':[],'num_detections':0,'all_detections':[]}

        msg_out      = String()
        msg_out.data = json.dumps(payload)
        self.pred_publisher.publish(msg_out)


def main(args=None):
    rclpy.init(args=args)
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path',  required=True)
    parser.add_argument('--class-map',   required=True)
    parser.add_argument('--conf-thresh', type=float, default=0.70)
    parsed, _ = parser.parse_known_args()

    node = RobotClassifierNode(
        parsed.model_path, parsed.class_map, parsed.conf_thresh)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
