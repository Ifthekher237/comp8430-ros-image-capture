#!/usr/bin/env python3
# ============================================================
# webcam_test.py
# Test the trained 3-class model on your laptop webcam.
# Uses OpenCV contour detection to find object regions in the
# frame, crops each ROI, classifies it, and draws bounding
# boxes with label + confidence.
#
# Press 'q' to quit.
# Press 's' to save current frame to outputs/predictions/.
#
# Usage:
#   python webcam_test.py \
#     --model-path outputs/models/three_class_robot_model.pth \
#     --class-map  outputs/models/three_class_class_to_idx.json
# ============================================================

import os
import sys
import argparse
import time
import yaml
import cv2
import torch

sys.path.insert(0, os.path.dirname(__file__))
from utils.model_utils    import load_checkpoint, get_device
from utils.inference_utils import (get_inference_transform,
                                   detect_and_classify, draw_detections)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Webcam test — multi-object ROI detection + classification")
    parser.add_argument('--model-path',  required=True,
                        help='Path to three_class_robot_model.pth')
    parser.add_argument('--class-map',   required=True,
                        help='Path to three_class_class_to_idx.json')
    parser.add_argument('--camera-id',   type=int, default=0,
                        help='Webcam device index (default: 0)')
    parser.add_argument('--conf-thresh', type=float, default=0.75,
                        help='Confidence threshold to display a prediction')
    parser.add_argument('--config',      default='config.yaml')
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = {}
    if os.path.exists(args.config):
        with open(args.config) as f:
            cfg = yaml.safe_load(f)

    os.makedirs(cfg.get('pred_dir', 'outputs/predictions'), exist_ok=True)

    device = get_device()

    # ---- Load model ----
    print(f"\nLoading model: {args.model_path}")
    model, class_to_idx, idx_to_class = load_checkpoint(
        args.model_path, device=str(device))

    transform = get_inference_transform(cfg.get('image_size', 224))

    # ROI detection params from config
    roi_params = {
        'blur_kernel':    cfg.get('roi_blur_kernel',    7),
        'canny_low':      cfg.get('roi_canny_low',     30),
        'canny_high':     cfg.get('roi_canny_high',   100),
        'dilate_iters':   cfg.get('roi_dilate_iters',   3),
        'min_area':       cfg.get('roi_min_area',    4000),
        'max_area_ratio': cfg.get('roi_max_area_ratio', 0.85),
        'padding':        cfg.get('roi_padding',       10),
    }

    conf_thresh = args.conf_thresh

    # ---- Open webcam ----
    print(f"Opening camera {args.camera_id} ...")
    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        print(f"ERROR: Could not open camera {args.camera_id}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\nWebcam running.")
    print("  Green box  = confidence above threshold")
    print("  Orange box = low confidence detection")
    print("  Press 'q' to quit | 's' to save frame\n")

    frame_count  = 0
    fps_timer    = time.time()
    fps_display  = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame — exiting.")
            break

        frame_count += 1

        # ---- Detect + classify ----
        detections = detect_and_classify(
            frame_bgr=frame,
            model=model,
            idx_to_class=idx_to_class,
            transform=transform,
            device=device,
            confidence_threshold=conf_thresh,
            **roi_params,
        )

        # ---- Draw results ----
        annotated = draw_detections(frame, detections, conf_thresh)

        # FPS counter (update every 30 frames)
        if frame_count % 30 == 0:
            elapsed = time.time() - fps_timer
            fps_display = 30.0 / elapsed if elapsed > 0 else 0
            fps_timer = time.time()

        # HUD overlay
        n_detected = sum(1 for d in detections if d['above_threshold'])
        hud = (f"FPS: {fps_display:.1f}  |  "
               f"Objects detected: {n_detected}  |  "
               f"Threshold: {conf_thresh:.2f}  |  "
               f"q=quit  s=save")
        cv2.putText(annotated, hud, (10, annotated.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imshow("Phase 3 — Webcam Test (ROI Detection)", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Quit.")
            break
        elif key == ord('s'):
            ts = int(time.time())
            save_path = os.path.join(
                cfg.get('pred_dir', 'outputs/predictions'),
                f"webcam_frame_{ts}.jpg")
            cv2.imwrite(save_path, annotated)
            print(f"  Frame saved: {save_path}")

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam closed.")


if __name__ == '__main__':
    main()
