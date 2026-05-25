# Robot Operations — Uni Day

---

## Before anything — every new terminal needs this

```bash
source ~/48548219/comp8430_phase3/install/setup.zsh
MODEL=~/48548219/comp8430_phase3/src/robot_classifier/robot_classifier
```

---

## SETUP — Once only (first time only)

**Upload `robot_deploy.zip` via NoMachine drag and drop.**

Then open terminal on Ubuntu:

```bash
cd ~/
unzip robot_deploy.zip -d ~/phase3_files
mkdir -p ~/phase3

cp ~/phase3_files/outputs/models/all_class_model.pth          ~/phase3/
cp ~/phase3_files/outputs/models/all_class_class_to_idx.json  ~/phase3/
cp ~/phase3_files/utils/inference_utils.py                    ~/phase3/
cp ~/phase3_files/ros2_nodes/robot_classifier_node.py         ~/phase3/
cp ~/phase3_files/ros2_nodes/camera_classifier_action_node.py ~/phase3/
cp ~/phase3_files/ros2_ws_setup/setup.py                      ~/phase3/
cp ~/phase3_files/ros2_ws_setup/package.xml                   ~/phase3/
```

```bash
mkdir -p ~/48548219/comp8430_phase3/src
cd ~/48548219/comp8430_phase3/src
ros2 pkg create robot_classifier --build-type ament_python --dependencies rclpy
```

```bash
PKG=~/48548219/comp8430_phase3/src/robot_classifier/robot_classifier

cp ~/phase3/robot_classifier_node.py          $PKG/
cp ~/phase3/camera_classifier_action_node.py  $PKG/
cp ~/phase3/inference_utils.py                $PKG/
cp ~/phase3/all_class_model.pth               $PKG/
cp ~/phase3/all_class_class_to_idx.json       $PKG/
cp ~/phase3/setup.py    ~/48548219/comp8430_phase3/src/robot_classifier/
cp ~/phase3/package.xml ~/48548219/comp8430_phase3/src/robot_classifier/
```

```bash
pip install ultralytics

cd ~/48548219/comp8430_phase3
colcon build
source install/setup.zsh
```

**Setup done. Never repeat.**

---

## DEMO 1 — Camera only, robot sits still

**Terminal 1:**
```bash
ros2 launch peripherals depth_camera.launch.py
```

**Terminal 2:**
```bash
source ~/48548219/comp8430_phase3/install/setup.zsh
MODEL=~/48548219/comp8430_phase3/src/robot_classifier/robot_classifier

ros2 run robot_classifier robot_classifier \
  --model-path $MODEL/all_class_model.pth \
  --class-map  $MODEL/all_class_class_to_idx.json
```

**Ctrl+C to stop.**

---

## DEMO 2 — Robot rotates and classifies everything

**Terminal 1:**
```bash
ros2 launch peripherals depth_camera.launch.py
```

**Terminal 2:**
```bash
ros2 launch bringup bringup.launch.py
```

**Terminal 3:**
```bash
source ~/48548219/comp8430_phase3/install/setup.zsh
MODEL=~/48548219/comp8430_phase3/src/robot_classifier/robot_classifier

ros2 run robot_classifier robot_demo \
  --model-path $MODEL/all_class_model.pth \
  --class-map  $MODEL/all_class_class_to_idx.json \
  --mode scan
```

**Ctrl+C to stop. Robot stops immediately.**

---

## DEMO 3 — Find one can and go to it

**Terminal 1 and 2 same as Demo 2. Keep them running.**

**Terminal 3 — pick target:**

```bash
source ~/48548219/comp8430_phase3/install/setup.zsh
MODEL=~/48548219/comp8430_phase3/src/robot_classifier/robot_classifier
```

**Cocacola_classic → moves forward:**
```bash
ros2 run robot_classifier robot_demo \
  --model-path $MODEL/all_class_model.pth \
  --class-map  $MODEL/all_class_class_to_idx.json \
  --mode target \
  --target-class Cocacola_classic
```

**Sprite → rotates left:**
```bash
ros2 run robot_classifier robot_demo \
  --model-path $MODEL/all_class_model.pth \
  --class-map  $MODEL/all_class_class_to_idx.json \
  --mode target \
  --target-class Sprite
```

**Fanta_orange → rotates right:**
```bash
ros2 run robot_classifier robot_demo \
  --model-path $MODEL/all_class_model.pth \
  --class-map  $MODEL/all_class_class_to_idx.json \
  --mode target \
  --target-class Fanta_orange
```

**Ctrl+C anytime to stop immediately.**

---

## If something goes wrong

| Problem | Fix |
|---------|-----|
| Node not found | `source ~/48548219/comp8430_phase3/install/setup.zsh` |
| Robot not moving | Check Terminal 2 has bringup running |
| cv_bridge error | `sudo apt install ros-humble-cv-bridge` |
| ultralytics error | `pip install ultralytics` |
