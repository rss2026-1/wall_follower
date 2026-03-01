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
        self.get_logger().info(f"Starting node")

        ##################
        # PID constants
        ##################
        self.kp = 1
        self.kd = 0.25
        self.k_angle = 0.8
        self.prev_error = 0.0
        self.prev_time = self.get_clock().now().nanoseconds / 1e9
        self.WALL_TOPIC = "/wall"
        self.line_pub = self.create_publisher(Marker, self.WALL_TOPIC, 1)

        # TODO: Initialize your publishers and subscribers here
        # Subscriber to LaserScan
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10)

        # Publisher to drive topic
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped,
            '/drive',
            10)

        self.get_logger().info("NEW VERSION RUNNING - 2")
        self.get_logger().info("Wall follower node started")

        # TODO: Write your callback functions here    

    def scan_callback(self, msg):
        ranges = np.array(msg.ranges)
        angles = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment

        # Choose side dynamically
        if self.SIDE == -1:   # RIGHT wall
            mask = ((angles < np.deg2rad(-60)) & (angles > np.deg2rad(-100))) | ((angles > np.deg2rad(-20)) & (angles < np.deg2rad(10)))
            # mask = ((angles < np.deg2rad(-60)) & (angles > np.deg2rad(-100)))
        else:                # LEFT wall
            # mask = ((angles > np.deg2rad(50)) & (angles < np.deg2rad(100))) | ((angles > np.deg2rad(-10)) & (angles < np.deg2rad(30)))
            mask = ((angles > np.deg2rad(10)) & (angles < np.deg2rad(100)))
            # mask = ((angles > np.deg2rad(60)) & (angles < np.deg2rad(100))) 

        side_ranges = ranges[mask]
        side_angles = angles[mask]

        # Remove invalid
        valid = np.isfinite(side_ranges)
        side_ranges = side_ranges[valid]
        side_angles = side_angles[valid]

        # Remove far points 
        max_dist = 9
        close = side_ranges < max_dist
        side_ranges = side_ranges[close]
        side_angles = side_angles[close]

        if len(side_ranges) < 2:
            self.get_logger().warn("Not enough wall points")
            return

        # Convert to Cartesian
        x = side_ranges * np.cos(side_angles)
        y = side_ranges * np.sin(side_angles)

        # Only forward-facing points
        # forward = x > 0.0
        # x = x[forward]
        # y = y[forward]

        if len(x) < 2:
            self.get_logger().warn("Not enough forward wall points")
            return

        # Least squares fit
        A = np.vstack([x, np.ones(len(x))]).T
        m, b = np.linalg.lstsq(A, y, rcond=None)[0]

        x = np.linspace(-10.0, 10.0, num=40)
        y = m*x+b
        VisualizationTools.plot_line(x, y, self.line_pub, frame="/laser")

        current_distance = b / np.sqrt(1 + m**2)
        error = self.DESIRED_DISTANCE - abs(current_distance)

        now = self.get_clock().now().nanoseconds / 1e9
        dt = now - self.prev_time if hasattr(self, "prev_time") else 0.01
        derivative = (error - self.prev_error) / dt if dt != 0 else 0
        self.prev_time = now
        self.prev_error = error

        if self.SIDE == -1:  # right wall
            steering = (self.kp * error + self.kd * derivative)
        else:               # left wall
            steering = -(self.kp * error + self.kd * derivative)
        steering = np.clip(steering, -0.34, 0.34) # Clamp steering

        # Publish
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = steering
        drive_msg.drive.speed = self.VELOCITY   # reduce speed when tuning
        # self.get_logger().info(f"SIDE: {self.SIDE}, dist: {current_distance:.3f}, error: {error:.3f}")
        self.drive_pub.publish(drive_msg)
                               
        """# Compute wall orientation
        theta_wall = np.arctan(m)

        # Distance error (always positive)
        distance_error = self.DESIRED_DISTANCE - abs(current_distance)

        # Time-based derivative
        now = self.get_clock().now().nanoseconds / 1e9
        dt = now - self.prev_time if hasattr(self, "prev_time") else 0.01
        self.prev_time = now

        if dt <= 0:
            dt = 0.01

        derivative = (distance_error - self.prev_error) / dt
        self.prev_error = distance_error

        control = (self.kp * distance_error + self.kd * derivative + self.k_angle * theta_wall)

        if self.SIDE == -1:      # RIGHT wall
            steering = control
        else:                    # LEFT wall
            steering = -control

        # Clamp steering
        steering = np.clip(steering, -0.34, 0.34)

        # Publish
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = steering
        drive_msg.drive.speed = self.VELOCITY
        self.drive_pub.publish(drive_msg)"""


    def parameters_callback(self, params):
        """
        DO NOT MODIFY THIS CALLBACK FUNCTION!
        
        This is used by the test cases to modify the parameters during testing. 
        It's called whenever a parameter is set via 'ros2 param set'.
        """
        # drive_msg = AckermannDriveStamped()
        # drive_msg.drive.steering_angle = 0.0
        # drive_msg.drive.speed = 0.0   # reduce speed when tuning

        # self.get_logger().info(f"SIDE: {self.SIDE}, dist: {current_distance:.3f}, error: {error:.3f}")

        # self.drive_pub.publish(drive_msg)

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
    