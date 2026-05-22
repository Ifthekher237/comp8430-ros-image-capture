#!/bin/bash
# Test 3 robot classes on laptop webcam — no ROS2, no robot needed
cd "$(dirname "$0")/.."
python webcam_test_all_classes.py \
  --model-path outputs/models/three_class_robot_model.pth \
  --class-map  outputs/models/three_class_class_to_idx.json
