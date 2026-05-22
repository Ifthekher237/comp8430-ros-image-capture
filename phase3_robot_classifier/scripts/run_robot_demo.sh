#!/bin/bash
# Run combined demo node — detect all 20 classes, navigate to target
# Change --target-class to any of the 20 class names
# source install/setup.zsh first
MODEL_PATH="$(pwd)/outputs/models/all_class_model.pth"
CLASS_MAP="$(pwd)/outputs/models/all_class_class_to_idx.json"
python ros2_nodes/camera_classifier_action_node.py \
  --model-path   "$MODEL_PATH" \
  --class-map    "$CLASS_MAP" \
  --target-class Redbull_Classic
