#!/usr/bin/env python3
# ============================================================
# robot_classifier_node.py
# DEMO 1 — Camera only. Robot does NOT move.
#
# Subscribes : /depth_cam/rgb/image_raw
# Publishes  : /object_classifier/prediction
#
# All code is self-contained — no utils/ dependency.
# Safe to copy as a single file into the ROS2 package.
#
# ros2 run robot_classifier robot_classifier \
#   --model-path /path/to/all_class_model.pth \
#   --class-map  /path/to/all_class_class_to_idx.json
# ============================================================

import sys, os, json, argparse
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
import cv2
import torch
import torch.nn.functional as F
from cv_bridge import CvBridge
from PIL import Image as PILImage
from torchvision import transforms, models
import torch.nn as nn

# ------------------------------------------------------------------
# YOLO — detects cans (bottle/cup class from COCO)
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
# Load MobileNetV3 checkpoint
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
# Detect cans + classify each crop
# ------------------------------------------------------------------
def _detect_and_classify(frame_bgr, model, idx_to_class,
                          transform, device, conf_thresh=0.70):
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
# Draw bounding boxes
# ------------------------------------------------------------------
def _draw(frame, detections, conf_thresh):
    out = frame.copy()
    for d in detections:
        x, y, w, h = d['bbox']
        above = d['above_threshold']
        color = (0, 200, 0) if above else (0, 140, 255)
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
# ROS2 Node
# ------------------------------------------------------------------
class RobotClassifierNode(Node):

    def __init__(self, model_path, conf_threshold=0.70):
        super().__init__('robot_classifier')

        self.cv_bridge      = CvBridge()
        self.conf_threshold = conf_threshold
        self.device         = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

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

        # Prediction publisher
        self.pred_publisher = self.create_publisher(
            String, '/object_classifier/prediction', 10)

        self.get_logger().info(
            f'Ready | {len(self.class_to_idx)} classes | '
            f'Robot does NOT move in this demo.')

    def image_callback(self, msg):
        image_bgr  = self.cv_bridge.imgmsg_to_cv2(msg, 'bgr8')
        detections = _detect_and_classify(
            image_bgr, self.model, self.idx_to_class,
            self.transform, self.device, self.conf_threshold)

        annotated = _draw(image_bgr, detections, self.conf_threshold)
        cv2.putText(annotated, 'DEMO 1 — Camera only | Robot stationary',
                    (10, annotated.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (200, 200, 200), 1, cv2.LINE_AA)
        cv2.imshow('Demo 1 — Camera Only', annotated)
        cv2.waitKey(1)

        above   = [d for d in detections if d['above_threshold']]
        best    = max(above, key=lambda d: d['confidence']) if above else None
        payload = ({
            'label':      best['label'],
            'confidence': round(best['confidence'], 4),
            'bbox':       list(best['bbox']),
        } if best else {'label': '', 'confidence': 0.0, 'bbox': []})

        msg_out      = String()
        msg_out.data = json.dumps(payload)
        self.pred_publisher.publish(msg_out)

        if best:
            self.get_logger().info(
                f"{best['label']}  conf={best['confidence']:.3f}")


def main(args=None):
    rclpy.init(args=args)
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path',  required=True)
    parser.add_argument('--class-map',   required=True)
    parser.add_argument('--conf-thresh', type=float, default=0.70)
    parsed, _ = parser.parse_known_args()

    node = RobotClassifierNode(parsed.model_path, parsed.conf_thresh)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()