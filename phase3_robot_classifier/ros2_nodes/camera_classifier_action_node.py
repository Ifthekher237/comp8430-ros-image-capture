#!/usr/bin/env python3
# ============================================================
# ros2_nodes/camera_classifier_action_node.py
# COMBINED demo node for Phase 3 final demo.
#
# Week 7  — CamVis subscriber + cv_bridge
# Week 8  — argparse parse_known_args
# Week 9  — Twist publisher on /cmd_vel
#
# THREE DISTINCT ACTIONS (one per deployment class):
#   Cocacola_classic → SEARCH + ALIGN + APPROACH (move forward)
#   Sprite           → SEARCH + ALIGN + ROTATE LEFT (circle)
#   Fanta_orange     → SEARCH + ALIGN + ROTATE RIGHT (circle)
#
# All 20 classes are detected and shown on screen.
# Only the 3 deployment classes trigger actions.
#
# Usage:
#   ros2 run robot_classifier robot_demo \
#     --model-path /path/to/robot_finetuned_model.pth \
#     --class-map  /path/to/robot_finetuned_class_to_idx.json \
#     --target-class Cocacola_classic
# ============================================================

import sys, os, json, argparse, time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from geometry_msgs.msg import Twist
import cv2
from cv_bridge import CvBridge
import torch

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_DIR, '..'))
sys.path.insert(0, _DIR)

from utils.model_utils     import load_checkpoint, get_device
from utils.inference_utils import (get_inference_transform,
                                   detect_and_classify, draw_detections)

# ---- Deployment classes and their actions ----
DEPLOYMENT_CLASSES = ['Cocacola_classic', 'Sprite', 'Fanta_orange']
CLASS_ACTIONS = {
    'Cocacola_classic': 'forward',        # move toward it
    'Sprite':           'rotate_left',    # circle left
    'Fanta_orange':     'rotate_right',   # circle right
}

# States
STATE_SEARCH   = 'SEARCH'
STATE_ALIGN    = 'ALIGN'
STATE_ACTION   = 'ACTION'    # class-specific action
STATE_ARRIVED  = 'ARRIVED'


class CameraClassifierActionNode(Node):

    def __init__(self, model_path, class_map_path,
                 target_class, conf_threshold=0.80,
                 consecutive_needed=5, frame_width=640):
        super().__init__('camera_classifier_action')

        # Validate target class
        if target_class not in DEPLOYMENT_CLASSES:
            raise ValueError(
                f'target_class must be one of {DEPLOYMENT_CLASSES}')

        self.target_class       = target_class
        self.target_action      = CLASS_ACTIONS[target_class]
        self.conf_threshold     = conf_threshold
        self.consecutive_needed = consecutive_needed
        self.frame_center       = frame_width / 2.0

        # Behaviour params
        self.search_angular_vel  = 0.3
        self.align_angular_gain  = 0.003
        self.approach_linear_vel = 0.1
        self.approach_ang_gain   = 0.001
        self.rotate_left_vel     = 0.5
        self.rotate_right_vel    = -0.5
        self.center_tolerance    = 80.0
        self.arrival_bbox_area   = 40000
        self.max_action_time     = 8.0

        self.state               = STATE_SEARCH
        self.consecutive_count   = 0
        self.action_start_time   = None

        # Load model — ALL classes
        self.device = get_device()
        self.model, self.class_to_idx, self.idx_to_class = \
            load_checkpoint(model_path, device=str(self.device))
        self.model.eval()
        self.transform = get_inference_transform(224)

        if target_class not in self.class_to_idx:
            self.get_logger().error(
                f'"{target_class}" not in model classes!')

        self.roi_params = {
            'blur_kernel':7, 'canny_low':30, 'canny_high':100,
            'dilate_iters':3, 'min_area':4000,
            'max_area_ratio':0.85, 'padding':10,
        }

        # cv_bridge — Week 7/8
        self.cv_bridge = CvBridge()

        # Camera subscriber — Week 7/8
        self.cam_subscription = self.create_subscription(
            Image, '/depth_cam/rgb/image_raw',
            self.image_callback, 1)

        # Velocity publisher — Week 9 Twist
        self.cmd_vel_publisher = self.create_publisher(
            Twist, '/cmd_vel', 10)

        # Prediction publisher
        self.pred_publisher = self.create_publisher(
            String, '/object_classifier/prediction', 10)

        self.get_logger().info(
            f'\n  CameraClassifierActionNode ready\n'
            f'  Classes    : {len(self.class_to_idx)}\n'
            f'  Target     : {self.target_class}\n'
            f'  Action     : {self.target_action}\n'
            f'  Threshold  : {self.conf_threshold}')

    # ------------------------------------------------------------------
    # Camera callback — Week 7 pattern
    # ------------------------------------------------------------------

    def image_callback(self, msg):
        image_bgr = self.cv_bridge.imgmsg_to_cv2(msg, 'bgr8')

        # Detect + classify all objects
        detections = detect_and_classify(
            image_bgr, self.model, self.idx_to_class,
            self.transform, self.device,
            self.conf_threshold, **self.roi_params)

        # Draw all bounding boxes
        annotated = draw_detections(image_bgr, detections, self.conf_threshold)

        # Overlay state info
        action_str = f"Action: {self.target_action}"
        cv2.putText(annotated,
                    f"State:{self.state} | Target:{self.target_class} | {action_str}",
                    (10, annotated.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (200, 200, 200), 1, cv2.LINE_AA)
        cv2.imshow('Phase 3 Demo', annotated)
        cv2.waitKey(1)

        # Find best detection of target
        target_dets = [d for d in detections
                       if d['label'] == self.target_class
                       and d['above_threshold']]
        target_visible = len(target_dets) > 0
        best = (max(target_dets, key=lambda d: d['confidence'])
                if target_visible else None)

        self._publish_prediction(best, detections)
        self._run_state_machine(target_visible, best)

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------

    def _run_state_machine(self, target_visible, best):

        # SEARCH — rotate slowly, scan for target
        if self.state == STATE_SEARCH:
            if target_visible:
                self.consecutive_count += 1
                self.get_logger().info(
                    f'[SEARCH] {self.target_class} '
                    f'conf={best["confidence"]:.3f} '
                    f'{self.consecutive_count}/{self.consecutive_needed}')
                if self.consecutive_count >= self.consecutive_needed:
                    self.state = STATE_ALIGN
                    self.consecutive_count = 0
                    self.get_logger().info('→ ALIGN')
            else:
                self.consecutive_count = 0
            self._publish_twist(0.0, self.search_angular_vel)

        # ALIGN — turn to centre target in frame
        elif self.state == STATE_ALIGN:
            if not target_visible:
                self.get_logger().info('Lost in ALIGN → SEARCH')
                self.state = STATE_SEARCH
                self.consecutive_count = 0
                self._publish_stop()
                return

            x, y, w, h = best['bbox']
            error = (x + w / 2.0) - self.frame_center

            if abs(error) <= self.center_tolerance:
                # Target centred — start class-specific action
                self.state = STATE_ACTION
                self.action_start_time = time.time()
                self.get_logger().info(
                    f'Centred → ACTION ({self.target_action})')
            else:
                angular_z = max(-0.4, min(0.4,
                                -self.align_angular_gain * error))
                self.get_logger().info(
                    f'[ALIGN] err={error:.1f}px ang={angular_z:.3f}')
                self._publish_twist(0.0, angular_z)

        # ACTION — class-specific behaviour
        elif self.state == STATE_ACTION:
            elapsed = time.time() - self.action_start_time

            if elapsed > self.max_action_time:
                self.get_logger().info('Action timeout → ARRIVED')
                self.state = STATE_ARRIVED
                self._publish_stop()
                return

            if not target_visible:
                self.get_logger().info('Lost in ACTION → SEARCH')
                self.state = STATE_SEARCH
                self.consecutive_count = 0
                self._publish_stop()
                return

            if self.target_action == 'forward':
                # Cocacola_classic — move forward toward it
                x, y, w, h = best['bbox']
                area  = w * h
                error = (x + w / 2.0) - self.frame_center
                if area >= self.arrival_bbox_area:
                    self.get_logger().info(
                        f'ARRIVED near {self.target_class}')
                    self.state = STATE_ARRIVED
                    self._publish_stop()
                    return
                angular_z = max(-0.15, min(0.15,
                                -self.approach_ang_gain * error))
                self.get_logger().info(
                    f'[FORWARD] area={area} err={error:.1f} t={elapsed:.1f}s')
                self._publish_twist(self.approach_linear_vel, angular_z)

            elif self.target_action == 'rotate_left':
                # Sprite — rotate left (circle around it)
                self.get_logger().info(
                    f'[ROTATE LEFT] t={elapsed:.1f}s')
                self._publish_twist(0.0, self.rotate_left_vel)

            elif self.target_action == 'rotate_right':
                # Fanta_orange — rotate right (circle around it)
                self.get_logger().info(
                    f'[ROTATE RIGHT] t={elapsed:.1f}s')
                self._publish_twist(0.0, self.rotate_right_vel)

        # ARRIVED
        elif self.state == STATE_ARRIVED:
            self._publish_stop()
            self.get_logger().info(
                f'[ARRIVED/DONE] Action complete for {self.target_class}')

    # ------------------------------------------------------------------
    # Twist helpers — Week 9 pattern
    # ------------------------------------------------------------------

    def _publish_twist(self, linear_x=0.0, angular_z=0.0):
        msg = Twist()
        msg.linear.x  = linear_x
        msg.linear.y  = 0.0; msg.linear.z  = 0.0
        msg.angular.x = 0.0; msg.angular.y = 0.0
        msg.angular.z = angular_z
        self.cmd_vel_publisher.publish(msg)

    def _publish_stop(self):
        self._publish_twist(0.0, 0.0)

    def _publish_prediction(self, best, all_dets):
        payload = ({
            'label':      best['label'],
            'confidence': round(best['confidence'], 4),
            'bbox':       list(best['bbox']),
            'action':     self.target_action,
        } if best else {
            'label':'','confidence':0.0,'bbox':[],'action':''
        })
        msg      = String()
        msg.data = json.dumps(payload)
        self.pred_publisher.publish(msg)


# ------------------------------------------------------------------
# main — Week 8 argparse pattern
# ------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)

    parser = argparse.ArgumentParser(
        description='Combined camera-classifier-action node')
    parser.add_argument('--model-path',   required=True)
    parser.add_argument('--class-map',    required=True)
    parser.add_argument('--target-class', required=True,
                        choices=DEPLOYMENT_CLASSES)
    parser.add_argument('--conf-thresh',  type=float, default=0.80)
    parser.add_argument('--consecutive',  type=int,   default=5)
    parser.add_argument('--frame-width',  type=int,   default=640)
    parsed, _ = parser.parse_known_args()

    node = CameraClassifierActionNode(
        model_path=parsed.model_path,
        class_map_path=parsed.class_map,
        target_class=parsed.target_class,
        conf_threshold=parsed.conf_thresh,
        consecutive_needed=parsed.consecutive,
        frame_width=parsed.frame_width,
    )
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
