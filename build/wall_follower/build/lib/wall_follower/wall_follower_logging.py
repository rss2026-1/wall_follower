#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker
from rcl_interfaces.msg import SetParametersResult
from std_msgs.msg import Float32

from wall_follower.visualization_tools import VisualizationTools


class WallFollowerLogging(Node):

    def __init__(self):
        super().__init__("wall_follower_logging")

        # Same parameters as the original node
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("drive_topic", "/vesc/low_level/input/navigation")
        self.declare_parameter("side", -1)
        self.declare_parameter("velocity", 1.0)
        self.declare_parameter("desired_distance", 1.0)

        # Fetch parameter values
        self.SCAN_TOPIC = self.get_parameter('scan_topic').get_parameter_value().string_value
        self.DRIVE_TOPIC = self.get_parameter('drive_topic').get_parameter_value().string_value
        self.SIDE = self.get_parameter('side').get_parameter_value().integer_value
        self.VELOCITY = self.get_parameter('velocity').get_parameter_value().double_value
        self.DESIRED_DISTANCE = self.get_parameter('desired_distance').get_parameter_value().double_value

        # Allow runtime parameter updates
        self.add_on_set_parameters_callback(self.parameters_callback)
        self.get_logger().info("Starting WallFollowerLogging node")

        # PID constants
        self.kp = 1.0
        self.kd = 0.25
        self.k_angle = 0.8
        self.prev_error = 0.0
        self.prev_time = self.get_clock().now().nanoseconds / 1e9

        # Visualization of the fitted wall line
        self.WALL_TOPIC = "/wall"
        self.line_pub = self.create_publisher(Marker, self.WALL_TOPIC, 1)

        # Subscriptions and publications
        self.scan_sub = self.create_subscription(
            LaserScan,
            self.SCAN_TOPIC,
            self.scan_callback,
            10,
        )

        self.drive_pub = self.create_publisher(
            AckermannDriveStamped,
            self.DRIVE_TOPIC,
            10,
        )

        # NEW: publishers for logging distance and error for later analysis
        self.distance_pub = self.create_publisher(Float32, "/wall/current_distance", 10)
        self.error_pub = self.create_publisher(Float32, "/wall/distance_error", 10)

        self.get_logger().info("WallFollowerLogging node initialized")

    def scan_callback(self, msg: LaserScan):
        # Convert ranges to numpy and build corresponding angle array
        ranges = np.array(msg.ranges)
        angles = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment

        # Same side-dependent masking strategy as the original node
        if self.SIDE == -1:  # RIGHT wall
            mask = (
                (angles < np.deg2rad(-30)) & (angles > np.deg2rad(-90))
            ) | (
                (angles > np.deg2rad(-20)) & (angles < np.deg2rad(20))
            )
        else:  # LEFT wall
            mask = (
                (angles > np.deg2rad(30)) & (angles < np.deg2rad(90))
            ) | (
                (angles > np.deg2rad(-20)) & (angles < np.deg2rad(20))
            )

        side_ranges = ranges[mask]
        side_angles = angles[mask]

        # Remove invalid (inf / NaN) entries
        valid = np.isfinite(side_ranges)
        side_ranges = side_ranges[valid]
        side_angles = side_angles[valid]

        # Discard very far points
        max_dist = 9.0
        close = side_ranges < max_dist
        side_ranges = side_ranges[close]
        side_angles = side_angles[close]

        if len(side_ranges) < 2:
            self.get_logger().warn("Not enough wall points")
            return

        # Polar to Cartesian in the laser frame
        x = side_ranges * np.cos(side_angles)
        y = side_ranges * np.sin(side_angles)

        if len(x) < 2:
            self.get_logger().warn("Not enough forward wall points")
            return

        # Least-squares fit y = m x + b
        A = np.vstack([x, np.ones(len(x))]).T
        m, b = np.linalg.lstsq(A, y, rcond=None)[0]

        # Visualize the fitted wall line in RViz
        x_vis = np.linspace(-10.0, 10.0, num=40)
        y_vis = m * x_vis + b
        VisualizationTools.plot_line(x_vis, y_vis, self.line_pub, frame="/laser")

        # Distance from laser (origin) to the fitted line
        current_distance = b / np.sqrt(1.0 + m ** 2)
        distance_error = self.DESIRED_DISTANCE - abs(current_distance)

        # Time-based derivative term
        now = self.get_clock().now().nanoseconds / 1e9
        dt = now - self.prev_time if self.prev_time is not None else 0.01
        if dt <= 0.0:
            dt = 0.01
        derivative = (distance_error - self.prev_error) / dt
        self.prev_time = now
        self.prev_error = distance_error

        # PD control (mirrored for left vs right wall)
        control = self.kp * distance_error + self.kd * derivative
        if self.SIDE == -1:  # right wall
            steering = control
        else:  # left wall
            steering = -control

        steering = float(np.clip(steering, -0.34, 0.34))

        # Publish drive command
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = steering
        drive_msg.drive.speed = float(self.VELOCITY)
        self.drive_pub.publish(drive_msg)

        # Publish logging signals for later plotting / analysis
        dist_msg = Float32()
        dist_msg.data = float(current_distance)
        self.distance_pub.publish(dist_msg)

        err_msg = Float32()
        err_msg.data = float(distance_error)
        self.error_pub.publish(err_msg)

    def parameters_callback(self, params):
        for param in params:
            if param.name == "side":
                self.SIDE = param.value
                self.get_logger().info(f"Updated side to {self.SIDE}")
            elif param.name == "velocity":
                self.VELOCITY = param.value
                self.get_logger().info(f"Updated velocity to {self.VELOCITY}")
            elif param.name == "desired_distance":
                self.DESIRED_DISTANCE = param.value
                self.get_logger().info(
                    f"Updated desired_distance to {self.DESIRED_DISTANCE}"
                )
        return SetParametersResult(successful=True)


def main():
    rclpy.init()
    node = WallFollowerLogging()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

