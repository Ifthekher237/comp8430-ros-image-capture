#!/bin/bash
cd "$(dirname "$0")/.."
python evaluate.py --data-dir ../dataset/robot_split/test --model-path outputs/models/robot_finetuned_model.pth --tag after_robot_finetune
