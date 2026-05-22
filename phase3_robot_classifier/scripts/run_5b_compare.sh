#!/bin/bash
cd "$(dirname "$0")/.."
python evaluate_before_after.py --test-dir ../dataset/robot_split/test --before-model outputs/models/all_class_model.pth --after-model outputs/models/robot_finetuned_model.pth
