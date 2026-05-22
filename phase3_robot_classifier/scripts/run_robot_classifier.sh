#!/bin/bash
# Run robot classifier on Ubuntu — uses ALL 20 classes
# source install/setup.zsh first
MODEL_PATH="$(pwd)/outputs/models/all_class_model.pth"
CLASS_MAP="$(pwd)/outputs/models/all_class_class_to_idx.json"
python ros2_nodes/robot_classifier_node.py \
  --model-path "$MODEL_PATH" \
  --class-map  "$CLASS_MAP"
