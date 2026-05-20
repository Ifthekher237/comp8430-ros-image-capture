import os
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import argparse


class CamCapture(Node):
    def __init__(self, name, save_path, interval, num_max):
        super().__init__(name)
        self.cam_subscription = self.create_subscription(Image, '/depth_cam/rgb/image_raw', self.image_callback, 1)
        self.cv_bridge = CvBridge()
        self.save_path = save_path
        self.capture_interval = float(interval)
        self.num_max = int(num_max)
        self.save_id = 1
        self.saved_count = 0
        self.last_capture_time = 0.0

        os.makedirs(self.save_path, exist_ok=True)

        self.get_logger().info(f"Selected save folder: {self.save_path}")
        self.get_logger().info(f"Capture interval: {self.capture_interval:.2f} seconds")
        self.get_logger().info(f"Maximum images to save: {self.num_max}")
        self.get_logger().info("Press 'q' in the preview window to stop capture.")

    def image_callback(self, msg):
        if msg is None:
            return

        try:
            image_bgr = self.cv_bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as exc:
            self.get_logger().error(f"Failed to convert ROS image: {exc}")
            return

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        cv2.imshow('capture rgb image', image_rgb)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.get_logger().info("'q' pressed, stopping capture.")
            rclpy.shutdown()
            return

        now = time.time()
        if now - self.last_capture_time < self.capture_interval:
            return

        image_name = f'image_{self.save_id:06d}.jpg'
        image_path = os.path.join(self.save_path, image_name)

        try:
            cv2.imwrite(image_path, image_bgr)
        except Exception as exc:
            self.get_logger().error(f"Failed to write image: {exc}")
            return

        self.get_logger().info(f"Saved image: {image_path}")
        self.last_capture_time = now
        self.save_id += 1
        self.saved_count += 1

        if self.num_max > 0 and self.saved_count >= self.num_max:
            self.get_logger().info(f"Reached num_max={self.num_max}, stopping capture.")
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    parser = argparse.ArgumentParser(description='Save images from ROS2 topic every interval seconds.')

    parser.add_argument(
        '--save_path',
        type=str,
        default='captured_images',
        help='Exact directory path to save captured images'
    )

    parser.add_argument(
        '--interval',
        type=float,
        default=2.0,
        help='Seconds between saved images'
    )

    parser.add_argument(
        '--num_max',
        type=int,
        default=100,
        help='Maximum number of images to save before stopping automatically'
    )

    parsed_args, unknown = parser.parse_known_args()

    node = CamCapture('capture_sub', parsed_args.save_path, parsed_args.interval, parsed_args.num_max)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Ctrl+C detected, shutting down.')
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main() 
