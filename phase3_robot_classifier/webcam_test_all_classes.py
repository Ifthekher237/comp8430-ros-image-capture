#!/usr/bin/env python3
# ============================================================
# webcam_test_all_classes.py
# Test classifier on LAPTOP WEBCAM. No ROS2. No robot.
# Works with any model (20-class or 3-class).
#
# Usage — all 20 classes:
#   python webcam_test_all_classes.py \
#     --model-path outputs/models/robot_finetuned_model.pth \
#     --class-map  outputs/models/robot_finetuned_class_to_idx.json
#
# Usage — 3 classes:
#   python webcam_test_all_classes.py \
#     --model-path outputs/models/three_class_robot_model.pth \
#     --class-map  outputs/models/three_class_class_to_idx.json
#
# Keys: q=quit  s=save frame  +=raise threshold  -=lower  d=debug
# ============================================================

import os, sys, argparse, time, yaml
import cv2, torch

sys.path.insert(0, os.path.dirname(__file__))
from utils.model_utils     import load_checkpoint, get_device
from utils.inference_utils import (get_inference_transform,
                                   detect_and_classify,
                                   draw_detections, detect_rois)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model-path',  default='outputs/models/robot_finetuned_model.pth')
    p.add_argument('--class-map',   default='outputs/models/robot_finetuned_class_to_idx.json')
    p.add_argument('--camera-id',   type=int,   default=0)
    p.add_argument('--conf-thresh', type=float, default=0.60)
    p.add_argument('--width',       type=int,   default=640)
    p.add_argument('--height',      type=int,   default=480)
    p.add_argument('--config',      default='config.yaml')
    return p.parse_args()


def draw_debug(frame, roi_params):
    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (roi_params['blur_kernel'],)*2, 0)
    edges   = cv2.Canny(blurred, roi_params['canny_low'], roi_params['canny_high'])
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edges, kernel, iterations=roi_params['dilate_iters'])
    cnts, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    debug   = frame.copy()
    cv2.drawContours(debug, cnts, -1, (0, 255, 255), 1)
    cv2.putText(debug, 'DEBUG — raw contours', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return debug


def main():
    args = parse_args()
    cfg  = yaml.safe_load(open(args.config)) if os.path.exists(args.config) else {}
    os.makedirs(cfg.get('pred_dir', 'outputs/predictions'), exist_ok=True)

    device = get_device()
    print(f'\nLoading: {args.model_path}')
    model, class_to_idx, idx_to_class = load_checkpoint(
        args.model_path, device=str(device))

    num_classes = len(class_to_idx)
    print(f'Classes loaded: {num_classes}')
    for name, idx in sorted(class_to_idx.items(), key=lambda x: x[1]):
        print(f'  [{idx:2d}] {name}')

    transform  = get_inference_transform(cfg.get('image_size', 224))
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
    debug_mode  = False

    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        print(f'ERROR: Camera {args.camera_id} not found. Try --camera-id 1')
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    print('\nWebcam running — q=quit  s=save  +=thresh up  -=thresh down  d=debug\n')

    frame_count = 0
    fps_timer   = time.time()
    fps_display = 0.0
    annotated   = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        if debug_mode:
            display = draw_debug(frame, roi_params)
        else:
            detections = detect_and_classify(
                frame, model, idx_to_class, transform, device,
                conf_thresh, **roi_params)
            annotated = draw_detections(frame, detections, conf_thresh)

            if frame_count % 30 == 0:
                elapsed     = time.time() - fps_timer
                fps_display = 30.0 / elapsed if elapsed > 0 else 0.0
                fps_timer   = time.time()

            above = [d for d in detections if d['above_threshold']]
            status = (f'FPS:{fps_display:.1f}  Model:{num_classes}cls  '
                      f'Thresh:{conf_thresh:.2f}  Det:{len(above)}  '
                      f'+/-thresh  d=debug  s=save  q=quit')
            cv2.rectangle(annotated, (0, 0),
                          (annotated.shape[1], 46), (30, 30, 30), cv2.FILLED)
            cv2.putText(annotated, status, (8, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (200, 200, 200), 1, cv2.LINE_AA)
            for i, d in enumerate(above[:10]):
                cv2.putText(annotated,
                            f"{d['label']} ({d['confidence']:.2f})",
                            (annotated.shape[1] - 270, 70 + i * 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                            (0, 230, 0), 1, cv2.LINE_AA)
            display = annotated

        cv2.imshow('Phase 3 — Laptop Webcam', display)
        key = cv2.waitKey(1) & 0xFF

        if   key == ord('q'):
            break
        elif key == ord('s') and annotated is not None:
            ts   = int(time.time())
            path = os.path.join('outputs/predictions', f'laptop_{ts}.jpg')
            cv2.imwrite(path, display)
            print(f'  Saved: {path}')
        elif key in (ord('+'), ord('=')):
            conf_thresh = min(0.99, round(conf_thresh + 0.05, 2))
            print(f'  Threshold → {conf_thresh}')
        elif key == ord('-'):
            conf_thresh = max(0.10, round(conf_thresh - 0.05, 2))
            print(f'  Threshold → {conf_thresh}')
        elif key == ord('d'):
            debug_mode = not debug_mode
            print(f'  Debug: {"ON" if debug_mode else "OFF"}')

    cap.release()
    cv2.destroyAllWindows()
    print('Done.')


if __name__ == '__main__':
    main()
