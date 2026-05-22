#!/usr/bin/env python3
# ============================================================
# ros2_nodes/robot_action_node.py
#
# Week 9 Move/Stop Twist publisher pattern.
#
# Subscribes to : /object_classifier/prediction (std_msgs/String)
# Publishes to  : /cmd_vel (geometry_msgs/Twist)
#
# Works with ALL 20 classes.
# --target-class can be ANY class from the dataset.
#
# Usage:
#   ros2 run robot_classifier robot_action \
#     --target-class Redbull_Classic
# ============================================================

import json
import argparse

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist

STATE_SEARCH   = 'SEARCH'
STATE_ALIGN    = 'ALIGN'
STATE_APPROACH = 'APPROACH'
STATE_ARRIVED  = 'ARRIVED'


class RobotActionNode(Node):
    """
    Subscribes to classifier predictions and publishes Twist
    velocity commands. Targets any class from the full 20-class model.
    Week 9 Twist publisher pattern.
    """

    def __init__(self, target_class: str,
                 conf_threshold: float = 0.80,
                 consecutive_needed: int = 5,
                 frame_width: int = 640):
        super().__init__('robot_action')

        self.target_class       = target_class
        self.conf_threshold     = conf_threshold
        self.consecutive_needed = consecutive_needed
        self.frame_center       = frame_width / 2.0

        # Behaviour parameters
        self.search_angular_vel  = 0.3
        self.align_angular_gain  = 0.003
        self.approach_linear_vel = 0.1
        self.approach_ang_gain   = 0.001
        self.center_tolerance    = 80.0
        self.arrival_bbox_area   = 40000
        self.max_approach_time   = 8.0

        self.state               = STATE_SEARCH
        self.consecutive_count   = 0
        self.approach_start_time = None

        # Subscriber — prediction from classifier node
        self.pred_subscription = self.create_subscription(
            String,
            '/object_classifier/prediction',
            self.prediction_callback,
            10)

        # Publisher — Week 9 Twist pattern
        self.cmd_vel_publisher = self.create_publisher(
            Twist, '/cmd_vel', 10)

        self.get_logger().info(
            f'RobotActionNode ready. '
            f'Target: "{self.target_class}" | '
            f'Threshold: {self.conf_threshold}')

    def prediction_callback(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warn('Bad prediction message — stopping.')
            self._publish_stop()
            return

        label = payload.get('label', '')
        conf  = payload.get('confidence', 0.0)
        bbox  = payload.get('bbox', [])

        target_visible = (label == self.target_class and
                          conf >= self.conf_threshold and
                          len(bbox) == 4)

        self._run_state_machine(target_visible, label, conf, bbox)

    def _run_state_machine(self, target_visible, label, conf, bbox):

        if self.state == STATE_SEARCH:
            if target_visible:
                self.consecutive_count += 1
                self.get_logger().info(
                    f'[SEARCH] {label} conf={conf:.3f} '
                    f'count={self.consecutive_count}/{self.consecutive_needed}')
                if self.consecutive_count >= self.consecutive_needed:
                    self.state = STATE_ALIGN
                    self.consecutive_count = 0
                    self.get_logger().info('→ ALIGN')
            else:
                self.consecutive_count = 0
            self._publish_twist(linear_x=0.0,
                                angular_z=self.search_angular_vel)

        elif self.state == STATE_ALIGN:
            if not target_visible:
                self.get_logger().info('Lost target in ALIGN → SEARCH')
                self.state = STATE_SEARCH
                self.consecutive_count = 0
                self._publish_stop()
                return

            x, y, w, h = bbox
            error = (x + w / 2.0) - self.frame_center

            if abs(error) <= self.center_tolerance:
                self.state = STATE_APPROACH
                self.approach_start_time = self.get_clock().now()
                self.get_logger().info(f'Centred (err={error:.1f}px) → APPROACH')
                self._publish_twist(linear_x=self.approach_linear_vel,
                                    angular_z=0.0)
            else:
                angular_z = max(-0.4, min(0.4,
                                -self.align_angular_gain * error))
                self.get_logger().info(
                    f'[ALIGN] err={error:.1f}px angular_z={angular_z:.3f}')
                self._publish_twist(linear_x=0.0, angular_z=angular_z)

        elif self.state == STATE_APPROACH:
            elapsed = (self.get_clock().now() -
                       self.approach_start_time).nanoseconds * 1e-9

            if elapsed > self.max_approach_time:
                self.get_logger().info('Max approach time → STOP')
                self.state = STATE_ARRIVED
                self._publish_stop()
                return

            if not target_visible:
                self.get_logger().info('Lost target in APPROACH → SEARCH')
                self.state = STATE_SEARCH
                self.consecutive_count = 0
                self._publish_stop()
                return

            x, y, w, h = bbox
            bbox_area = w * h
            error     = (x + w / 2.0) - self.frame_center

            if bbox_area >= self.arrival_bbox_area:
                self.get_logger().info(
                    f'ARRIVED near {label} (area={bbox_area}) → STOP')
                self.state = STATE_ARRIVED
                self._publish_stop()
                return

            angular_z = max(-0.15, min(0.15,
                            -self.approach_ang_gain * error))
            self.get_logger().info(
                f'[APPROACH] area={bbox_area} '
                f'err={error:.1f}px elapsed={elapsed:.1f}s')
            self._publish_twist(linear_x=self.approach_linear_vel,
                                angular_z=angular_z)

        elif self.state == STATE_ARRIVED:
            self._publish_stop()

    # Week 9 Twist helpers
    def _publish_twist(self, linear_x=0.0, angular_z=0.0):
        msg = Twist()
        msg.linear.x  = linear_x
        msg.linear.y  = 0.0
        msg.linear.z  = 0.0
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = angular_z
        self.cmd_vel_publisher.publish(msg)

    def _publish_stop(self):
        self._publish_twist(0.0, 0.0)


def main(args=None):
    rclpy.init(args=args)

    # Week 8 argparse pattern — target-class accepts any class name
    parser = argparse.ArgumentParser(
        description='Robot action node — navigate to any target class')
    parser.add_argument('--target-class',  type=str, required=True,
                        help='Any class name from the dataset e.g. Redbull_Classic')
    parser.add_argument('--conf-thresh',   type=float, default=0.80)
    parser.add_argument('--consecutive',   type=int,   default=5)
    parser.add_argument('--frame-width',   type=int,   default=640)
    parsed_args, unknown = parser.parse_known_args()

    node = RobotActionNode(
        target_class=parsed_args.target_class,
        conf_threshold=parsed_args.conf_thresh,
        consecutive_needed=parsed_args.consecutive,
        frame_width=parsed_args.frame_width,
    )

    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
