# COMP8430 Phase 3 — Robot Image Classifier

**Deployment classes:** Cocacola_classic · Sprite · Fanta_orange  
**Model:** MobileNetV3-Large (transfer learning)  
**Detection:** OpenCV ROI contour-based bounding boxes

---

## Project Structure

```
phase3_robot_classifier/
├── train_all_classes.py          STEP 1 — train all 20 classes (Phase 2 dataset)
├── prepare_robot_dataset.py      STEP 2 — split robot images into train/val/test
├── evaluate.py                   STEP 3 — evaluate before fine-tune
├── train_robot_finetune.py       STEP 4 — fine-tune on robot images (all 20 classes)
├── evaluate_before_after.py      STEP 5 — compare before vs after (one command)
├── train_three_classes.py        STEP 6 — final 3-class fine-tune for robot
├── webcam_test_all_classes.py    STEP 7 — test at home with laptop webcam
├── config.yaml                   central configuration
├── requirements.txt
├── utils/
│   ├── dataset_utils.py          splits, loaders, augmentation
│   ├── model_utils.py            build/save/load MobileNetV3
│   ├── eval_utils.py             metrics, confusion matrix, comparison plot
│   └── inference_utils.py        ROI detection + classification (shared)
├── ros2_nodes/
│   ├── robot_classifier_node.py  ROS2 classifier (Week 7/8 pattern)
│   └── camera_classifier_action_node.py  ROS2 combined demo (Week 9 pattern)
├── ros2_ws_setup/
│   ├── setup.py                  drop-in for Ubuntu robot ROS2 package
│   └── package.xml
├── notebooks/
│   └── phase3_training_evaluation.ipynb
├── scripts/                      numbered shell scripts for each step
└── outputs/
    ├── models/                   saved .pth checkpoints
    ├── logs/                     metrics JSON and CSV
    ├── plots/                    training curves, confusion matrices
    └── predictions/              per-image prediction CSV, saved webcam frames
```

---

## Installation

```bash
cd comp8430-ros-image-capture/phase3_robot_classifier
pip install -r requirements.txt
```

---

## LAPTOP PIPELINE — Run in Order

### STEP 1 — Train on all 20 classes (Phase 2 clean dataset)
```bash
python train_all_classes.py \
  --data-dir ../dataset/clean_dataset \
  --epochs 20 --batch-size 16
```
Outputs: `outputs/models/all_class_model.pth` + `all_class_class_to_idx.json`

---

### STEP 2 — Prepare robot-captured images
Place your robot-captured images here first:
```
../dataset/robot_captured/
    Cocacola_classic/   image0.jpg  image1.jpg ...
    Sprite/             ...
    Fanta_orange/       ...
    ... (all 20 classes)
```
Then run:
```bash
python prepare_robot_dataset.py \
  --source-dir ../dataset/robot_captured \
  --output-dir ../dataset/robot_split
```

---

### STEP 3 — Evaluate BEFORE robot fine-tuning
```bash
python evaluate.py \
  --data-dir   ../dataset/robot_split/test \
  --model-path outputs/models/all_class_model.pth \
  --tag        before_robot_finetune
```

---

### STEP 4 — Fine-tune on robot images (all 20 classes)
```bash
python train_robot_finetune.py \
  --data-dir   ../dataset/robot_split \
  --base-model outputs/models/all_class_model.pth \
  --epochs 20 --batch-size 16
```
Outputs: `outputs/models/robot_finetuned_model.pth`

---

### STEP 5 — Evaluate AFTER + comparison (one command)
```bash
python evaluate_before_after.py \
  --test-dir     ../dataset/robot_split/test \
  --before-model outputs/models/all_class_model.pth \
  --after-model  outputs/models/robot_finetuned_model.pth
```
Outputs: comparison CSV + per-class bar chart

---

### STEP 6 — Final 3-class fine-tune for robot deployment
```bash
python train_three_classes.py \
  --robot-dir  ../dataset/robot_split \
  --output-dir ../dataset/three_class_robot \
  --base-model outputs/models/robot_finetuned_model.pth \
  --epochs 15
```
Outputs: `outputs/models/three_class_robot_model.pth`

---

### STEP 7 — Test at home with laptop webcam
```bash
# All 20 classes
python webcam_test_all_classes.py \
  --model-path outputs/models/robot_finetuned_model.pth \
  --class-map  outputs/models/robot_finetuned_class_to_idx.json

# 3 classes only
python webcam_test_all_classes.py \
  --model-path outputs/models/three_class_robot_model.pth \
  --class-map  outputs/models/three_class_class_to_idx.json
```
Keys: `q`=quit · `s`=save frame · `+`/`-`=threshold · `d`=debug contours

---

## ROBOT DEPLOYMENT (Ubuntu Machine)

### Transfer to robot
```bash
scp -r outputs/models/          ubuntu@ROBOT_IP:~/phase3/
scp -r ros2_nodes/              ubuntu@ROBOT_IP:~/phase3/
scp    utils/inference_utils.py ubuntu@ROBOT_IP:~/phase3/
scp    ros2_ws_setup/setup.py   ubuntu@ROBOT_IP:~/phase3/
scp    ros2_ws_setup/package.xml ubuntu@ROBOT_IP:~/phase3/
```

### Create ROS2 package (Week 8 pattern)
```bash
mkdir -p ~/48548219/comp8430_phase3/src
cd ~/48548219/comp8430_phase3/src
ros2 pkg create robot_classifier --build-type ament_python --dependencies rclpy

PKG=~/48548219/comp8430_phase3/src/robot_classifier/robot_classifier
cp ~/phase3/ros2_nodes/*.py         $PKG/
cp ~/phase3/inference_utils.py      $PKG/
cp ~/phase3/models/robot_finetuned_model.pth          $PKG/
cp ~/phase3/models/robot_finetuned_class_to_idx.json  $PKG/
cp ~/phase3/setup.py    ~/48548219/comp8430_phase3/src/robot_classifier/
cp ~/phase3/package.xml ~/48548219/comp8430_phase3/src/robot_classifier/
```

### Build (Week 8/9 pattern)
```bash
cd ~/48548219/comp8430_phase3
colcon build
source install/setup.zsh
```

### Run the demo

Terminal 1:
```bash
ros2 launch peripherals depth_camera.launch.py
```

Terminal 2:
```bash
ros2 launch bringup bringup.launch.py
```

Terminal 3:
```bash
source ~/48548219/comp8430_phase3/install/setup.zsh
MODEL=~/48548219/comp8430_phase3/src/robot_classifier/robot_classifier

# Target: Cocacola_classic → robot moves FORWARD
ros2 run robot_classifier robot_demo \
  --model-path $MODEL/robot_finetuned_model.pth \
  --class-map  $MODEL/robot_finetuned_class_to_idx.json \
  --target-class Cocacola_classic

# Target: Sprite → robot ROTATES LEFT
ros2 run robot_classifier robot_demo \
  --model-path $MODEL/robot_finetuned_model.pth \
  --class-map  $MODEL/robot_finetuned_class_to_idx.json \
  --target-class Sprite

# Target: Fanta_orange → robot ROTATES RIGHT
ros2 run robot_classifier robot_demo \
  --model-path $MODEL/robot_finetuned_model.pth \
  --class-map  $MODEL/robot_finetuned_class_to_idx.json \
  --target-class Fanta_orange
```

---

## Robot Behaviour

| State | What happens | Velocities |
|-------|-------------|------------|
| SEARCH | Rotates slowly, scans all 20 classes | angular.z = 0.3 |
| ALIGN | Target seen 5 frames, turns to centre it | angular.z = proportional |
| ACTION (Cocacola_classic) | Moves forward toward object | linear.x = 0.1 |
| ACTION (Sprite) | Rotates left around object | angular.z = +0.5 |
| ACTION (Fanta_orange) | Rotates right around object | angular.z = -0.5 |
| ARRIVED/DONE | Stops | all zero |

---

## Troubleshooting

**Camera not opening:** Try `--camera-id 1`

**No ROIs detected:** Lower `roi_min_area` in `config.yaml` (try 2000). Ensure object has contrast against background.

**ROS2 node not found:** Run `source install/setup.zsh` in every new terminal.

**Low confidence:** Lower `--conf-thresh` to 0.60. Or press `-` key in webcam test.

**cv_bridge error on Ubuntu:** `sudo apt install ros-humble-cv-bridge`
