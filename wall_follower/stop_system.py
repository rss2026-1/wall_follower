import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped

class StopSystem(Node):
    def __init__(self):
        super().__init__("stop_system")
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("drive_topic", "/vesc/low_level/ackermann_cmd")
        self.declare_parameter("stop_topic", "/vesc/low_level/input/safety")

        self.SCAN_TOPIC = self.get_parameter('scan_topic').get_parameter_value().string_value
        self.DRIVE_TOPIC = self.get_parameter('drive_topic').get_parameter_value().string_value

        self.min_distance_threshold = 0.5
        self.min_dist = float('inf')

        self.scan_subscriber = self.create_subscription(
            LaserScan,
            self.SCAN_TOPIC,
            self.scan_callback,
            10
        )
        self.drive_subcriber= self.create_subscription(
            AckermannDriveStamped,
            self.DRIVE_TOPIC,
            self.drive_callback,
            10
        )

        self.stop_publisher = self.create_publisher(
            AckermannDriveStamped,
            self.get_parameter('stop_topic').get_parameter_value().string_value,
            10
        )

    def scan_callback(self, msg):
        x = msg.ranges * np.cos(np.arange(len(msg.ranges)) * msg.angle_increment + msg.angle_min)




    def drive_callback(self, msg):
        

    def publish_stop(self):
        stop_msg = AckermannDriveStamped()
        stop_msg.drive.speed = 0.0
        stop_msg.drive.steering_angle = 0.0
        self.stop_publisher.publish(stop_msg)
