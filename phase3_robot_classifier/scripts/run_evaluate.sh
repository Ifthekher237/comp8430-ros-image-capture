#!/bin/bash
cd "$(dirname "$0")/.."
python evaluate.py \
  --data-dir ../dataset/phase3_three_classes/test \
  --model-path outputs/models/three_class_robot_model.pth \
  --tag three_class_eval
