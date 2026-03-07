from math import dist

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped

class StopSystem(Node):
    def __init__(self):
        super().__init__("stop_system")
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("vel_topic", "/vesc/odom")
        self.declare_parameter("stop_topic", "/vesc/low_level/input/safety")

        self.SCAN_TOPIC = self.get_parameter('scan_topic').get_parameter_value().string_value
        self.VEL_TOPIC = self.get_parameter('vel_topic').get_parameter_value().string_value

        self.safety_margin = 0.2   # minimum buffer
        self.max_decel = 1.5       # max deceleration
        self.min_dist = float('inf')

        self.velocity = 0.0

        self.scan_subscriber = self.create_subscription(
            LaserScan,
            self.SCAN_TOPIC,
            self.scan_callback,
            10
        )
        self.drive_subscriber = self.create_subscription(
            AckermannDriveStamped,
            self.VEL_TOPIC,
            self.vel_callback,
            10
        )

        self.stop_publisher = self.create_publisher(
            AckermannDriveStamped,
            self.get_parameter('stop_topic').get_parameter_value().string_value,
            10
        )

    def scan_callback(self, msg):
        ranges = np.array(msg.ranges)
        angles = np.arange(len(ranges)) * msg.angle_increment + msg.angle_min


        # Consider only a frontal cone
        frontal_mask = np.abs(angles) < 0.25
        frontal_ranges = ranges[frontal_mask]

        self.min_dist = np.min(frontal_ranges)
        # turning_distance = 0.5/np.sin(0.34)

        stop_dist = self.get_stopping_distance(self.velocity)

        stop_dist = 0.
        if self.min_dist < stop_dist:
            self.publish_stop()

    def get_stopping_distance(self, vel):
        # Compute stopping distance
        dist = 0.137 * max(abs(vel), 0.0) ** 2.48 + self.safety_margin

        # Log the value (do NOT call get_stopping_distance again)
        self._logger.info(f"Current velocity: {vel:.2f} m/s, Stopping distance: {dist:.2f} m, Min distance: {self.min_dist:.2f} m")

        return dist

    def drive_callback(self, msg):
        self.velocity = msg.drive.speed

    def publish_stop(self):
        stop_msg = AckermannDriveStamped()
        stop_msg.drive.speed = 0.0
        stop_msg.drive.steering_angle = 0.0
        self.stop_publisher.publish(stop_msg)

def main():
    rclpy.init()
    stop_system = StopSystem()
    rclpy.spin(stop_system)
    stop_system.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
