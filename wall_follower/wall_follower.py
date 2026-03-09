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

class WallFollower(Node):

    def __init__(self):

        super().__init__("wall_follower")
        # Declare parameters to make them available for use
        # DO NOT MODIFY THIS!
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("drive_topic", "/vesc/low_level/input/navigation")
        # self.declare_parameter("drive_topic", "/drive")
        self.declare_parameter("side", -1)
        self.declare_parameter("velocity", 1.5)
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
        self.kp = 2.0
        self.kd = 0.5
        # self.k_angle = 0.8

        # Lookahead
        self.k_lookahead = 1            # gain on predicted future error
        self.LOOKAHEAD_DIST = 2.0       # how many meters to look ahead

        # RANSAC
        self.RANSAC_ITERATIONS  = 40    # number of random trials
        self.RANSAC_SAMPLE_SIZE = 2     # points per trial (min for a line)
        self.RANSAC_THRESHOLD   = 0.08  # inlier tolerance (metres)
        self.RANSAC_MIN_INLIERS = 6     # reject fit if fewer inliers than this

        # Time
        self.prev_error = 0.0
        self.prev_time = self.get_clock().now().nanoseconds / 1e9

        self.WALL_TOPIC = "/wall"
        self.line_pub = self.create_publisher(Marker, self.WALL_TOPIC, 20)

        # TODO: Initialize your publishers and subscribers here
        # Subscriber to LaserScan
        self.scan_sub = self.create_subscription(
            LaserScan,
            self.SCAN_TOPIC,
            self.scan_callback,
            10)

        # Publisher to drive topic
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped,
            self.DRIVE_TOPIC,
            10)

        self.scan_dist_pub = self.create_publisher(
            Float32,
            "/scan_dist",
            10
        )

        self.steering_pub = self.create_publisher(
            Float32,
            "/steering",
            10
        )


        self.get_logger().info("NEW VERSION RUNNING - KDF")
        self.get_logger().info("Wall follower node started")

    # TODO: Write your callback functions here

    def _extract_wall_points(self, ranges, angles):
        '''
        Return (x,y) coordinates of the wall being tracked.

        The bands based on angle
            1. side_zone - perpendicular to wall
            2. forward_zone - forward facing for lookahead (turning capability)

        Bounded by MAX_DIST; discarding inf/NaN values
        '''
        MAX_DIST = 5.0

        if self.SIDE == -1:   # RIGHT wall
            side_mask = (angles >= np.deg2rad(-90)) & (angles <= np.deg2rad(-30))
            forward_mask = (angles >  np.deg2rad(-30)) & (angles <= np.deg2rad(-10))
        else:                # LEFT wall
            side_mask = (angles >= np.deg2rad(30)) & (angles <= np.deg2rad(90))
            forward_mask = (angles >=  np.deg2rad(10)) & (angles < np.deg2rad(30))

        def to_xy(mask):
            r = ranges[mask]
            a = angles[mask]
            valid = np.isfinite(r) & (r < MAX_DIST) #& (r > 0.05)
            x = r[valid] * np.cos(a[valid])
            y = r[valid] * np.sin(a[valid])
            return x, y

        side_x, side_y = to_xy(side_mask)
        forward_x, forward_y = to_xy(forward_mask)

        return side_x, side_y, forward_x, forward_y

    def _ransac_line_fit(self,x,y):
        '''
        RANSAC line of best fit

        Returns y=mx+b line of best fit. (m,b,inlier_mask)
        '''

        best_inliers = None
        best_count = self.RANSAC_MIN_INLIERS - 1

        n = len(x)
        if n < self.RANSAC_SAMPLE_SIZE:
            return None, None, None

        for _ in range(self.RANSAC_ITERATIONS):
            # Min random sample
            idx = np.random.choice(n, self.RANSAC_SAMPLE_SIZE, replace=False)
            x_s, y_s = x[idx], y[idx]

            dx = x_s[1] - x_s[0]
            if abs(dx) < 1e-6:
                continue # skip vertical lines in x
            m_try = (y_s[1] - y_s[0]) / dx
            b_try = y_s[0] - m_try * x_s[0]

            # Perpendicular distance of all points to candidate line
            # |m*xi - yi + b| / sqrt(m^2+1)
            dist = np.abs(m_try * x - y + b_try) / np.sqrt(m_try**2 + 1)
            inliers = dist < self.RANSAC_THRESHOLD

            if inliers.sum() > best_count:
                best_count   = inliers.sum()
                best_inliers = inliers

        if best_inliers is None:
            return None, None, None

        # Refit with inliers for better estimate
        xi, yi = x[best_inliers], y[best_inliers]
        A = np.vstack([xi, np.ones(len(xi))]).T
        m, b = np.linalg.lstsq(A, yi, rcond=None)[0]

        return m, b, best_inliers

    def _perpendicular_distance(self, m, b):
        return b / np.sqrt(1.00+m**2)

    def _lookahead_error(self,m,b):
        L = self.LOOKAHEAD_DIST
        d_ahead = (m*L+b) / np.sqrt(1.0 + m **2)
        return self.DESIRED_DISTANCE - abs(d_ahead)

    def scan_callback(self,msg):
        '''
        1. Extract wall points from scan
        2. RANSAC for line of best fit
        3. Compute error
        4. PD control
        5. Publish
        '''
        # Get ranges and angles
        ranges = np.array(msg.ranges)
        angles = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment

        # Extract wall points from scan
        side_x, side_y, forward_x, forward_y = self._extract_wall_points(ranges, angles)
        # combine all wall points for regression
        x_all = np.concatenate([side_x, forward_x])
        y_all = np.concatenate([side_y, forward_y])

        if len(x_all) < self.RANSAC_SAMPLE_SIZE:
            self.get_logger().warn("Not enough wall points for RANSAC")
            return

        # RANSAC
        # m, b, inliers = self._ransac_line_fit(x_all, y_all)

        # Lienar regression
        A = np.vstack([x_all, np.ones(len(x_all))]).T
        m, b = np.linalg.lstsq(A, y_all, rcond=None)[0]

        if m is None:
            self.get_logger().warn("RANSAC failed- could not find good wall fit")
            return

        # Visualizer
        x_vis = np.linspace(-10.0, 10.0, 40)
        y_vis = m*x_vis+b
        VisualizationTools.plot_line(x_vis, y_vis, self.line_pub)
        dist_current = abs(self._perpendicular_distance(m,b))

        # Detect potential second wall
        # m2, b2, inliers2 = self._ransac_line_fit(x_all[~inliers], y_all[~inliers])

        # if m2 is not None:
        #     y_vis2 = m2*x_vis+b2
        #     VisualizationTools.plot_line(x_vis, y_vis2, self.line_pub)
        #     dist_current2 = abs(self._perpendicular_distance(m2,b2))
        #     if dist_current2 <= dist_current:
        #         dist_current, m, b = dist_current2, m2, b2


        # Compute error
        error_current = self.DESIRED_DISTANCE - dist_current
        error_ahead = self._lookahead_error(m,b)

        # Derivative
        now = self.get_clock().now().nanoseconds / 1e9
        dt = now - self.prev_time if hasattr(self, "prev_time") else 0.01
        derivative = (error_current - self.prev_error) / dt if dt != 0 else 0
        self.prev_time = now
        self.prev_error = error_current

        # PD Controller
        control = (self.kp * error_current
                    + self.kd * derivative
                    + self.k_lookahead * error_ahead)

        # Adjust for sign convention
        if self.SIDE == -1: # right wall
            steering = control
        else: #left wall
            steering = -control

        steering = float(np.clip(steering, -0.34, 0.34)) # Clamp steering based on given bounds

        scan_dist_msg = Float32()
        scan_dist_msg.data = float(error_current)
        self.scan_dist_pub.publish(scan_dist_msg)

        steering_msg = Float32()
        steering_msg.data = float(steering)
        self.steering_pub.publish(steering_msg)

        # Publish
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = steering
        drive_msg.drive.speed = self.VELOCITY   # reduce speed when tuning
        self.drive_pub.publish(drive_msg)
        # self.get_logger().info("Current distance: " + str(dist_current))
        # self.get_logger().info("Error ahead: " + str(steering))

    '''
    def scan_callback(self, msg):
        ranges = np.array(msg.ranges)
        angles = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment

        # Choose side dynamically

        side_ranges = ranges[mask]
        side_angles = angles[mask]

        # Remove invalid
        valid = np.isfinite(side_ranges)
        side_ranges = side_ranges[valid]
        side_angles = side_angles[valid]

        # Remove far points
        max_dist = 5
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
        error = self.DESIRED_DISTANCE - (abs(current_distance) + np.cos(np.arctan(m)) * 0.5) #add look ahead factor of distance to forward facing wall multiplied by angle between wall and car to make the car steer better around corners

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
        self.get_logger().info("Current diatance: " + str(current_distance))

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
    '''

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
