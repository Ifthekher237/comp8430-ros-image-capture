#!/bin/bash
cd "$(dirname "$0")/.."
python train_robot_finetune.py --data-dir ../dataset/robot_split --base-model outputs/models/all_class_model.pth --epochs 20 --batch-size 16
