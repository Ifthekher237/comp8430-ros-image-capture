#!/bin/bash
cd "$(dirname "$0")/.."
python train_three_classes.py --robot-dir ../dataset/robot_split --output-dir ../dataset/three_class_robot --base-model outputs/models/robot_finetuned_model.pth --epochs 15
