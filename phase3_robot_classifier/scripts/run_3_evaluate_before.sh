#!/bin/bash
cd "$(dirname "$0")/.."
python evaluate.py --data-dir ../dataset/robot_split/test --model-path outputs/models/all_class_model.pth --tag before_robot_finetune
