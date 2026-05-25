#!/usr/bin/env python3
# ============================================================
# camera_classifier_action_node.py
#
# DEMO 2 — SCAN mode:
#   Robot rotates slowly, detects and classifies all cans.
#   Ctrl+C to stop.
#
# DEMO 3 — TARGET mode:
#   Robot rotates, finds the target can, moves FORWARD toward it.
#   Stops when it gets close enough.
#   Works for ALL 3 classes: Cocacola_classic, Sprite, Fanta_orange.
#
# Week 7  — CamVis subscriber + cv_bridge
# Week 8  — argparse parse_known_args
# Week 9  — Twist publisher on /cmd_vel
#
# DEMO 2:
#   ros2 run robot_classifier robot_demo \
#     --model-path ... --class-map ... --mode scan
#
# DEMO 3:
#   ros2 run robot_classifier robot_demo \
#     --model-path ... --class-map ... \
#     --mode target --target-class Cocacola_classic
#
#   ros2 run robot_classifier robot_demo \
#     --model-path ... --class-map ... \
#     --mode target --target-class Sprite
#
#   ros2 run robot_classifier robot_demo \
#     --model-path ... --class-map ... \
#     --mode target --target-class Fanta_orange
# ============================================================

import sys, os, json, argparse, time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from geometry_msgs.msg import Twist
import cv2
import torch
import torch.nn.functional as F
from cv_bridge import CvBridge
from PIL import Image as PILImage
from torchvision import transforms, models
import torch.nn as nn

# ------------------------------------------------------------------
# YOLO
# ------------------------------------------------------------------
_yolo_model = None
CAN_CLASSES = {'bottle', 'cup', 'wine glass', 'vase', 'sports ball', 'bowl'}

def _get_yolo():
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        _yolo_model = YOLO('yolov8n.pt')
    return _yolo_model

# ------------------------------------------------------------------
# Transform
# ------------------------------------------------------------------
def _get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

# ------------------------------------------------------------------
# Load model
# ------------------------------------------------------------------
def _load_model(model_path, device):
    ckpt         = torch.load(model_path, map_location=device)
    num_classes  = ckpt.get('num_classes')
    class_to_idx = ckpt.get('class_to_idx', {})
    idx_to_class = {int(v): k for k, v in class_to_idx.items()}
    model = models.mobilenet_v3_large(weights=None)
    in_f  = model.classifier[3].in_features
    model.classifier[3] = nn.Sequential(
        nn.Dropout(p=0.3), nn.Linear(in_f, num_classes))
    model.load_state_dict(ckpt['state_dict'])
    model.to(device).eval()
    return model, class_to_idx, idx_to_class

# ------------------------------------------------------------------
# Detect + classify
# ------------------------------------------------------------------
def _detect_and_classify(frame_bgr, model, idx_to_class,
                          transform, device, conf_thresh=0.75):
    yolo = _get_yolo()
    h, w = frame_bgr.shape[:2]
    results = yolo(frame_bgr, verbose=False, conf=0.15)[0]
    detections = []
    for box in results.boxes:
        if results.names[int(box.cls[0])] not in CAN_CLASSES:
            continue
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x1 = max(0, int(x1) - 8);  y1 = max(0, int(y1) - 8)
        x2 = min(w, int(x2) + 8);  y2 = min(h, int(y2) + 8)
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        rgb    = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor = transform(PILImage.fromarray(rgb)).unsqueeze(0).to(device)
        with torch.no_grad():
            probs      = F.softmax(model(tensor), dim=1)
            conf, pred = probs.max(1)
        label = idx_to_class[pred.item()]
        c     = conf.item()
        detections.append({
            'bbox':            (x1, y1, x2 - x1, y2 - y1),
            'label':           label,
            'confidence':      c,
            'above_threshold': c >= conf_thresh,
        })
    return detections

# ------------------------------------------------------------------
# Draw
# ------------------------------------------------------------------
def _draw(frame, detections, conf_thresh):
    out = frame.copy()
    for d in detections:
        x, y, w, h = d['bbox']
        color = (0, 200, 0) if d['above_threshold'] else (0, 140, 255)
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        text = f"{d['label']}  {d['confidence']:.2f}"
        (tw, th), bl = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        ly = max(y - 6, th + 6)
        cv2.rectangle(out,
                      (x, ly - th - bl - 4),
                      (x + tw + 6, ly + bl - 2),
                      color, cv2.FILLED)
        cv2.putText(out, text, (x + 3, ly - bl - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return out

# ------------------------------------------------------------------
# Deployment classes
# ------------------------------------------------------------------
DEPLOYMENT_CLASSES = ['Cocacola_classic', 'Sprite', 'Fanta_orange']

STATE_SEARCH  = 'SEARCH'
STATE_ALIGN   = 'ALIGN'
STATE_APPROACH = 'APPROACH'
STATE_ARRIVED = 'ARRIVED'

# ------------------------------------------------------------------
# ROS2 Node
# ------------------------------------------------------------------
class CameraClassifierActionNode(Node):

    def __init__(self, model_path, mode='scan',
                 target_class=None, conf_threshold=0.75,
                 consecutive_needed=5, frame_width=640):
        super().__init__('camera_classifier_action')

        self.mode               = mode
        self.target_class       = target_class
        self.conf_threshold     = conf_threshold
        self.consecutive_needed = consecutive_needed
        self.frame_center       = frame_width / 2.0

        # Movement params
        self.search_angular_vel  = 0.3    # rotate while searching
        self.align_angular_gain  = 0.003  # proportional turn to centre can
        self.approach_linear_vel = 0.1    # move forward toward can
        self.approach_ang_gain   = 0.001  # small correction while moving
        self.center_tolerance    = 80.0   # pixels — dead zone
        self.arrival_bbox_area   = 40000  # stop when can fills this area
        self.max_approach_time   = 8.0    # safety stop after 8 seconds

        self.state             = STATE_SEARCH
        self.consecutive_count = 0
        self.approach_start_time = None

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.cv_bridge = CvBridge()

        self.get_logger().info('Loading model ...')
        self.model, self.class_to_idx, self.idx_to_class = \
            _load_model(model_path, self.device)
        self.transform = _get_transform()

        self.get_logger().info('Loading YOLOv8n ...')
        _get_yolo()

        # Camera subscriber — Week 7/8 pattern
        self.cam_subscription = self.create_subscription(
            Image, '/depth_cam/rgb/image_raw',
            self.image_callback, 1)

        # Velocity publisher — Week 9 Twist pattern
        self.cmd_vel_publisher = self.create_publisher(
            Twist, '/cmd_vel', 10)

        # Prediction publisher
        self.pred_publisher = self.create_publisher(
            String, '/object_classifier/prediction', 10)

        if self.mode == 'scan':
            self.get_logger().info(
                '\n  DEMO 2 — SCAN MODE\n'
                '  Robot rotates slowly.\n'
                '  Detects and classifies every can it sees.\n'
                '  Ctrl+C to stop immediately.')
        else:
            self.get_logger().info(
                f'\n  DEMO 3 — TARGET MODE\n'
                f'  Target : {self.target_class}\n'
                f'  Robot will rotate → find can → move forward → stop.')

    # ------------------------------------------------------------------
    # Camera callback — Week 7 pattern
    # ------------------------------------------------------------------
    def image_callback(self, msg):
        image_bgr  = self.cv_bridge.imgmsg_to_cv2(msg, 'bgr8')
        detections = _detect_and_classify(
            image_bgr, self.model, self.idx_to_class,
            self.transform, self.device, self.conf_threshold)

        annotated = _draw(image_bgr, detections, self.conf_threshold)

        if self.mode == 'scan':
            label_text = 'DEMO 2 — SCAN | rotating and classifying'
        else:
            label_text = (f'DEMO 3 | Target: {self.target_class} '
                          f'| State: {self.state}')
        cv2.putText(annotated, label_text,
                    (10, annotated.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (200, 200, 200), 1, cv2.LINE_AA)
        cv2.imshow('Phase 3 Robot Demo', annotated)
        cv2.waitKey(1)

        above = [d for d in detections if d['above_threshold']]
        best  = max(above, key=lambda d: d['confidence']) if above else None
        self._publish_prediction(best)

        if self.mode == 'scan':
            self._scan_mode()
        else:
            self._target_mode(detections)

    # ------------------------------------------------------------------
    # DEMO 2 — just rotate and classify, no target
    # ------------------------------------------------------------------
    def _scan_mode(self):
        self._publish_twist(0.0, self.search_angular_vel)

    # ------------------------------------------------------------------
    # DEMO 3 — find target, align, move forward toward it
    # ALL 3 classes move forward — same behaviour for all
    # ------------------------------------------------------------------
    def _target_mode(self, detections):
        target_dets    = [d for d in detections
                          if d['label'] == self.target_class
                          and d['above_threshold']]
        target_visible = len(target_dets) > 0
        best           = (max(target_dets, key=lambda d: d['confidence'])
                          if target_visible else None)

        # SEARCH — rotate and look for target
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

        # ALIGN — turn until can is centred in frame
        elif self.state == STATE_ALIGN:
            if not target_visible:
                self.get_logger().info('Lost → back to SEARCH')
                self.state = STATE_SEARCH
                self.consecutive_count = 0
                self._publish_stop()
                return
            x, y, w, h = best['bbox']
            error = (x + w / 2.0) - self.frame_center
            if abs(error) <= self.center_tolerance:
                # Can is centred — start moving forward
                self.state = STATE_APPROACH
                self.approach_start_time = time.time()
                self.get_logger().info(
                    f'Can centred → APPROACH (moving forward)')
            else:
                # Turn proportionally to centre the can
                az = max(-0.4, min(0.4,
                         -self.align_angular_gain * error))
                self.get_logger().info(
                    f'[ALIGN] error={error:.1f}px  turning={az:.3f}')
                self._publish_twist(0.0, az)

        # APPROACH — move forward toward the can
        # Same for ALL 3 classes — Cocacola, Sprite, Fanta all move forward
        elif self.state == STATE_APPROACH:
            elapsed = time.time() - self.approach_start_time

            # Safety stop after max time
            if elapsed > self.max_approach_time:
                self.get_logger().info(
                    f'Max approach time reached → STOP')
                self.state = STATE_ARRIVED
                self._publish_stop()
                return

            # Lost the can — go back to search
            if not target_visible:
                self.get_logger().info('Lost can → back to SEARCH')
                self.state = STATE_SEARCH
                self.consecutive_count = 0
                self._publish_stop()
                return

            x, y, w, h = best['bbox']
            area  = w * h
            error = (x + w / 2.0) - self.frame_center

            # Arrived — bounding box is large enough = robot is close
            if area >= self.arrival_bbox_area:
                self.get_logger().info(
                    f'ARRIVED near {self.target_class} — STOP')
                self.state = STATE_ARRIVED
                self._publish_stop()
                return

            # Move forward + small correction to stay centred on can
            az = max(-0.15, min(0.15,
                     -self.approach_ang_gain * error))
            self.get_logger().info(
                f'[APPROACH] moving forward | '
                f'bbox_area={area} | '
                f'elapsed={elapsed:.1f}s')
            self._publish_twist(self.approach_linear_vel, az)

        # ARRIVED — stop everything
        elif self.state == STATE_ARRIVED:
            self._publish_stop()
            self.get_logger().info(
                f'DONE — stopped near {self.target_class}')

    # ------------------------------------------------------------------
    # Twist helpers — Week 9 pattern
    # ------------------------------------------------------------------
    def _publish_twist(self, linear_x=0.0, angular_z=0.0):
        msg           = Twist()
        msg.linear.x  = linear_x
        msg.linear.y  = 0.0
        msg.linear.z  = 0.0
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = angular_z
        self.cmd_vel_publisher.publish(msg)

    def _publish_stop(self):
        self._publish_twist(0.0, 0.0)

    def _publish_prediction(self, best):
        payload = ({
            'label':      best['label'],
            'confidence': round(best['confidence'], 4),
            'bbox':       list(best['bbox']),
        } if best else {'label': '', 'confidence': 0.0, 'bbox': []})
        msg      = String()
        msg.data = json.dumps(payload)
        self.pred_publisher.publish(msg)


# ------------------------------------------------------------------
# main — Week 8 argparse pattern
# ------------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path',   required=True)
    parser.add_argument('--class-map',    required=True)
    parser.add_argument('--mode',         default='scan',
                        choices=['scan', 'target'])
    parser.add_argument('--target-class', default=None,
                        choices=DEPLOYMENT_CLASSES)
    parser.add_argument('--conf-thresh',  type=float, default=0.75)
    parser.add_argument('--consecutive',  type=int,   default=5)
    parser.add_argument('--frame-width',  type=int,   default=640)
    parsed, _ = parser.parse_known_args()

    if parsed.mode == 'target' and parsed.target_class is None:
        print('ERROR: --target-class required when --mode target')
        print('Choose: Cocacola_classic | Sprite | Fanta_orange')
        return

    node = CameraClassifierActionNode(
        model_path=parsed.model_path,
        mode=parsed.mode,
        target_class=parsed.target_class,
        conf_threshold=parsed.conf_thresh,
        consecutive_needed=parsed.consecutive,
        frame_width=parsed.frame_width,
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node._publish_stop()
        node.get_logger().info('Stopped by user.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()