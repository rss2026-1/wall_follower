#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker
from rcl_interfaces.msg import SetParametersResult

from wall_follower.visualization_tools import VisualizationTools


class WallFollower(Node):
    """
    Wall-following controller for the simulated racecar.

    High‑level behavior:
      1. Listen to the lidar scan on SCAN_TOPIC.
      2. Keep only points on the side we care about (left/right) and in front.
      3. Estimate how far we are from that wall using the *average sideways distance*
         of those points (matches the autograder’s metric).
      4. Run a PD controller on that distance error to choose a steering angle.
      5. Drive forward at the commanded velocity, slowing down a bit when the
         distance error is large so we don’t overshoot in tight turns.
    """

    def __init__(self):
        super().__init__("wall_follower")
        # Declare parameters to make them available for use
        # DO NOT MODIFY THIS!
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("drive_topic", "/drive")
        self.declare_parameter("side", 1)
        self.declare_parameter("velocity", 1.0)
        self.declare_parameter("desired_distance", 1.0)

        # Fetch constants from the ROS parameter server
        # DO NOT MODIFY THIS! This is necessary for the tests to be able to test varying parameters!
        self.SCAN_TOPIC = self.get_parameter('scan_topic').get_parameter_value().string_value
        self.DRIVE_TOPIC = self.get_parameter('drive_topic').get_parameter_value().string_value
        self.SIDE = self.get_parameter('side').get_parameter_value().integer_value
        self.VELOCITY = self.get_parameter('velocity').get_parameter_value().double_value
        self.DESIRED_DISTANCE = self.get_parameter('desired_distance').get_parameter_value().double_value

        # This activates the parameters_callback function so that the tests are able
        # to change the parameters during testing.
        # DO NOT MODIFY THIS!
        self.add_on_set_parameters_callback(self.parameters_callback)

        # Initialize publishers and subscribers
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, self.DRIVE_TOPIC, 1
        )
        self.scan_sub = self.create_subscription(
            LaserScan, self.SCAN_TOPIC, self.scan_callback, 1
        )

        #Controller configuration and internal state
        self.prev_error = 0.0
        self.prev_time = None
        self.Kp = 1.2
        self.Kd = 0.15
        self.max_steering_angle = 0.34 # radians
        self.x_min = 0.15   # min forward distance
        self.x_max = 2.0    # max forward distance
        self.range_min = 0.1
        self.range_max = 10.0
        self.slow_down_error_threshold = 0.4

    def scan_callback(self, msg):
        """Process LaserScan: filter points on the chosen side, estimate distance, PD control, publish drive"""
        # Build angles for each range (robot frame: 0 = +x forward, +angle = left)
        ranges = np.array(msg.ranges, dtype=np.float64)
        n = len(ranges)
        angles = msg.angle_min + np.arange(n, dtype=np.float64) * msg.angle_increment
        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)

        # Select the lidar points that are useful for wall following
        valid_range = np.isfinite(ranges) & (ranges >= self.range_min) & (ranges <= self.range_max)
        correct_side = self.SIDE * y > 0
        in_front = (x >= self.x_min) & (x <= self.x_max)
        valid = valid_range & correct_side & in_front

        xs = x[valid]
        ys = y[valid]

        if xs.size < 5:
            drive = AckermannDriveStamped()
            drive.drive.speed = self.VELOCITY * 0.5
            drive.drive.steering_angle = 0.0
            self.drive_pub.publish(drive)
            return

        current_distance = float(np.mean(np.abs(ys)))
        error = current_distance - self.DESIRED_DISTANCE

        # Derivative term
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.prev_time is not None:
            dt = now - self.prev_time
            if dt > 0:
                error_dot = (error - self.prev_error) / dt
            else:
                error_dot = 0.0
        else:
            error_dot = 0.0
        self.prev_error = error
        self.prev_time = now

        # PD: steer toward wall when too far, away when too close. SIDE: +1 left, -1 right
        steering = self.SIDE * (self.Kp * error + self.Kd * error_dot)
        steering = np.clip(steering, -self.max_steering_angle, self.max_steering_angle)
        speed = float(self.VELOCITY)
        if abs(error) > self.slow_down_error_threshold:
            speed = 0.6 * self.VELOCITY

        drive = AckermannDriveStamped()
        drive.drive.speed = speed
        drive.drive.steering_angle = float(steering)
        self.drive_pub.publish(drive)

    def parameters_callback(self, params):
        """
        DO NOT MODIFY THIS CALLBACK FUNCTION!

        This is used by the test cases to modify the parameters during testing.
        It's called whenever a parameter is set via 'ros2 param set'.
        """
        for param in params:
            if param.name == 'side':
                self.SIDE = param.value
                self.get_logger().info(f"Updated side to {self.SIDE}")
            elif param.name == 'velocity':
                self.VELOCITY = param.value
                self.get_logger().info(f"Updated velocity to {self.VELOCITY}")
            elif param.name == 'desired_distance':
                self.DESIRED_DISTANCE = param.value
                self.get_logger().info(f"Updated desired_distance to {self.DESIRED_DISTANCE}")
        return SetParametersResult(successful=True)


def main():
    rclpy.init()
    wall_follower = WallFollower()
    rclpy.spin(wall_follower)
    wall_follower.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
