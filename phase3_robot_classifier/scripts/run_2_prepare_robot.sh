#!/bin/bash
cd "$(dirname "$0")/.."
python prepare_robot_dataset.py --source-dir ../dataset/robot_captured --output-dir ../dataset/robot_split
