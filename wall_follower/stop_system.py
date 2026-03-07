import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped

class StopSystem(Node):
    def __init__(self):
        super().__init__("stop_system")
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("vel_topic", "/vesc/odom")
        self.declare_parameter("stop_topic", "/vesc/low_level/input/safety")

        self.SCAN_TOPIC = self.get_parameter('scan_topic').get_parameter_value().string_value
        self.VEL_TOPIC  = self.get_parameter('vel_topic').get_parameter_value().string_value

        self.safety_margin = 0.30  # increased from 0.2 to account for odom snap artifact
        self.min_dist = float('inf')
        self.velocity = 0.0

        self.scan_subscriber = self.create_subscription(
            LaserScan, self.SCAN_TOPIC, self.scan_callback, 10)

        self.vel_subscriber = self.create_subscription(
            Odometry, self.VEL_TOPIC, self.vel_callback, 10)

        self.stop_publisher = self.create_publisher(
            AckermannDriveStamped,
            self.get_parameter('stop_topic').get_parameter_value().string_value,
            10)

    def scan_callback(self, msg):
        ranges = np.array(msg.ranges)
        angles = np.arange(len(ranges)) * msg.angle_increment + msg.angle_min

        frontal_mask = np.abs(angles) < 0.25
        self.min_dist = np.min(ranges[frontal_mask])

        stop_dist = self.get_stopping_distance(self.velocity)  # no overwrite!

        if self.min_dist < stop_dist:
            self.publish_stop()

    def get_stopping_distance(self, vel):
        # Validated against log: 1.0m/s→0.21m, 1.5m/s→0.45m, 2.0m/s→0.74m
        d = 0.210 * max(abs(vel), 0.0) ** 1.825 + self.safety_margin
        self.get_logger().info(
            f"Current velocity: {vel:.2f} m/s, "
            f"Stopping distance: {d:.2f} m, "
            f"Min distance: {self.min_dist:.2f} m")
        return d

    def vel_callback(self, msg):
        self.velocity = msg.twist.twist.linear.x

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
