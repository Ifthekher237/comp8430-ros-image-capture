#!/bin/bash
cd "$(dirname "$0")/.."
python prepare_three_class_dataset.py \
  --source-dir ../dataset/clean_dataset \
  --output-dir ../dataset/phase3_three_classes \
  --classes Redbull_Classic Redbull_zero Oxyshred_Passion
