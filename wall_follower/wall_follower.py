#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker
from rcl_interfaces.msg import SetParametersResult
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import LaserScan

from wall_follower.visualization_tools import VisualizationTools

class WallFollower(Node):

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

        # Publishers / subscribers
        self.drive_pub = self.create_publisher(AckermannDriveStamped, self.DRIVE_TOPIC, 1)
        self.wall_pub = self.create_publisher(Marker, '/wall', 1)
        self.scan_sub = self.create_subscription(LaserScan, self.SCAN_TOPIC, self.scan_callback, 1)

        # Controller gains (tweakable)
        self.Kp = 1.0
        self.Kd = 0.25
        self.k_angle = 0.6  # small orientation term

        # Limits and filters
        self.max_steering_angle = 0.34
        self.x_min = 0.05
        self.x_max = 4.0
        self.range_min = 0.05
        self.range_max = 10.0
        self.slow_down_error_threshold = 0.6

        # State for derivative and filtering
        self.prev_error = 0.0
        self.prev_time = None
        self.derivative = 0.0
        self.derivative_alpha = 0.2  # exponential smoothing factor for derivative

        # Minimum dt to avoid extreme derivatives
        self.min_dt = 1e-3
        self.max_dt = 0.5

    def scan_callback(self, msg: LaserScan):
        """Process LaserScan: select side, fit line, compute PD control, publish drive."""
        ranges = np.array(msg.ranges, dtype=np.float64)
        n = len(ranges)
        angles = msg.angle_min + np.arange(n, dtype=np.float64) * msg.angle_increment

        # Cartesian coordinates in robot frame
        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)

        # Valid point mask
        valid_range = np.isfinite(ranges) & (ranges >= self.range_min) & (ranges <= self.range_max)
        correct_side = self.SIDE * y > 0
        in_front = (x >= self.x_min) & (x <= self.x_max)
        valid = valid_range & correct_side & in_front

        xs = x[valid]
        ys = y[valid]

        # Fallback if not enough points: slow down and go straight (or small bias)
        if xs.size < 6:
            drive = AckermannDriveStamped()
            drive.drive.speed = float(self.VELOCITY * 0.5)
            drive.drive.steering_angle = 0.0
            self.drive_pub.publish(drive)
            return

        # Least-squares fit of wall: y = m*x + b (x is forward)
        A = np.vstack([xs, np.ones_like(xs)]).T
        try:
            m, b = np.linalg.lstsq(A, ys, rcond=None)[0]
        except Exception:
            # numerical fallback if least-squares fails- ends function
            drive = AckermannDriveStamped()
            drive.drive.speed = float(self.VELOCITY * 0.5)
            drive.drive.steering_angle = 0.0
            self.drive_pub.publish(drive)
            return

        # Visualization of fitted line (for debugging/rviz)
        xv = np.linspace(0.0, 6.0, num=20)
        yv = m * xv + b
        VisualizationTools.plot_line(xv, yv, self.wall_pub, frame="/laser")

        # Distance from origin (robot) to line
        current_distance = abs(b) / np.sqrt(1.0 + m ** 2)
        # Wall orientation
        theta_wall = np.arctan(m)

        # Distance error: positive if too far from wall
        error = current_distance - self.DESIRED_DISTANCE

        # Time & derivative
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.prev_time is not None:
            dt = now - self.prev_time
            if dt <= 0:
                dt = self.min_dt
        else:
            dt = 0.05

        # Clamp steering with derivative values to avoid spikes 
        dt = float(np.clip(dt, self.min_dt, self.max_dt))

        raw_derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        # Exponential smoothing on derivative to reduce noise (self.derivative_alpha term)
        self.derivative = (1.0 - self.derivative_alpha) * self.derivative + self.derivative_alpha * raw_derivative

        # Updating time and error
        self.prev_error = error
        self.prev_time = now

        # PD + small orientation term
        control = self.Kp * error + self.Kd * self.derivative + self.k_angle * theta_wall

        # SIDE: +1 left, -1 right — apply sign so positive control steers toward wall when too far
        steering = self.SIDE * control
        steering = float(np.clip(steering, -self.max_steering_angle, self.max_steering_angle))

        # Speed scheduling: slow down for large errors
        speed = float(self.VELOCITY)
        if abs(error) > self.slow_down_error_threshold:
            speed = 0.6 * self.VELOCITY

        # Publish drive
        drive = AckermannDriveStamped()
        drive.drive.speed = speed
        drive.drive.steering_angle = steering
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
    