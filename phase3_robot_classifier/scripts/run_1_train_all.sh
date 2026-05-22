#!/bin/bash
cd "$(dirname "$0")/.."
python train_all_classes.py --data-dir ../dataset/clean_dataset --epochs 20 --batch-size 16
