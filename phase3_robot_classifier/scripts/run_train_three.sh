#!/bin/bash
cd "$(dirname "$0")/.."
python train_three_classes.py \
  --data-dir ../dataset/phase3_three_classes \
  --epochs 15 --batch-size 16 \
  --base-model outputs/models/all_class_model.pth
