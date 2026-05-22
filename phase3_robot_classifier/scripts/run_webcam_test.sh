#!/bin/bash
cd "$(dirname "$0")/.."
python webcam_test.py \
  --model-path outputs/models/three_class_robot_model.pth \
  --class-map  outputs/models/three_class_class_to_idx.json
