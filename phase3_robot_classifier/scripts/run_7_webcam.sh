#!/bin/bash
cd "$(dirname "$0")/.."
python webcam_test_all_classes.py --model-path outputs/models/robot_finetuned_model.pth --class-map outputs/models/robot_finetuned_class_to_idx.json
