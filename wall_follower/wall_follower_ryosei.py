
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

        # your publishers and subscribers here
        self.scan_subscriber = self.create_subscription(
            LaserScan,
            self.SCAN_TOPIC,
            self.scan_callback,
            10
        )
        self.drive_publisher = self.create_publisher(
            AckermannDriveStamped,
            self.DRIVE_TOPIC,
            10
        )

        self.prev_dist_error = 0.0
        self.prev_angle_error = 0.0

        # Clamp steering to max physical range
        self.max_steer = 0.34  # ~20 degrees

    def scan_callback(self, msg):
        dists = np.array(msg.ranges)
        angles = np.linspace(msg.angle_min, msg.angle_max, len(msg.ranges))

        #convert to cartesian
        x = dists * np.cos(angles)
        y = dists * np.sin(angles)

        if self.SIDE == 1:  # left
            side_mask = y > 0
        else:               # right
            side_mask = y < 0

        #mask unneccesary scan data
        forward_mask = (x > 0) & (x < 4.0)
        mask = side_mask & forward_mask

        X = x[mask]
        Y = y[mask]

        # Check for wall ahead
        front_candidates = [i for i, j in zip(x, y) if i > 0 and abs(j) < 0.5]
        front_dist = min(front_candidates) if front_candidates else float('inf')

        # Check if there's a wall on the followed side
        # If very few points or they're all very far, opening exists
        has_wall = len(X) > 10 and np.median(np.abs(y[mask])) < 3.0

        front_threshold = 0.5 + self.VELOCITY * 0.3
        if self.VELOCITY ==2.0:
            front_threshold = 1.3

        if not has_wall:
            steering = self.SIDE * self.max_steer
            self.prev_dist_error = 0.0
        elif front_dist < front_threshold:
            steering = -self.SIDE * self.max_steer
            self.prev_dist_error = 0.0
        else:
            # Normal wall following with PD controller
            # Use OLS for regression
            X = np.column_stack([X, np.ones_like(X)])
            th = np.linalg.inv(X.T @ X) @ X.T @ Y

            # Distance from origin to fitted line y = slope*x + intercept
            wall_dist = np.abs(th[1]) / np.sqrt(th[0] ** 2 + 1)

            # Compute errors (always, so they're defined in both branches)
            dist_error = (self.DESIRED_DISTANCE - wall_dist) * -self.SIDE

            Kp_dist = 1.5
            Kd_dist = 2.0

            steering = Kp_dist * dist_error + Kd_dist * (dist_error - self.prev_dist_error)
            self.prev_dist_error = dist_error

        steering = np.clip(steering, -self.max_steer, self.max_steer)

        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.drive.steering_angle = float(steering)
        drive_msg.drive.speed = self.VELOCITY
        self.drive_publisher.publish(drive_msg)


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

