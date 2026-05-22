#!/bin/bash
# Test ALL 20 classes on laptop webcam — no ROS2, no robot needed
cd "$(dirname "$0")/.."
python webcam_test_all_classes.py \
  --model-path outputs/models/all_class_model.pth \
  --class-map  outputs/models/all_class_class_to_idx.json
