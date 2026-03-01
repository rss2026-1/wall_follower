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
  
        self.publisher_ = self.create_publisher(AckermannDriveStamped, self.DRIVE_TOPIC, 20)
        self.publisher_wall = self.create_publisher(Marker, 'wall', 20)
        self.publisher_path = self.create_publisher(Marker, 'path', 20)
        # self.publisher_test = self.create_publisher(Float32, 'wall', 20)
        self.subscription = self.create_subscription(
            LaserScan,
            self.SCAN_TOPIC,
            self.listener_callback,
            20)
        self.subscription
        self.start = 27
        self.end = 38

    def listener_callback(self, msg):
        time = self.get_clock().now().to_msg()

        angle_increment = msg.angle_increment
        if self.SIDE == -1:
            ranges = np.array(msg.ranges[self.start:self.end+1])
            angles = np.array([angle_increment*i-(3*np.pi/4) for i in range(self.start, self.end+1)])
        else:
            ranges = np.array(msg.ranges[-self.end-1:-self.start])
            angles = np.array([angle_increment*i+(3*np.pi/4) for i in range(-self.end, -self.start+1)])
        closest_idx = np.argmin(ranges)
        x_coord = ranges * np.cos(angles)
        y_coord = ranges * np.sin(angles)

        start = max(closest_idx-10, 0)
        end = start + 21

        sub_x = x_coord[start:end]
        sub_y = y_coord[start:end]

        sub_x = x_coord
        sub_y = y_coord

        oc = np.polyfit(sub_x, sub_y, 1)
        coeff = np.polyfit(sub_x, sub_y, 1)
        original_line = np.poly1d(oc)
        coeff[1] = (y_coord[closest_idx] - coeff[0]*x_coord[closest_idx]) - self.SIDE*(coeff[0]**2 + (self.DESIRED_DISTANCE*1.2))**0.5
        offset_line = np.poly1d(coeff)
        angle_offset = np.arctan(coeff)[0]


        VisualizationTools.plot_line(
            [0.0, x_coord[closest_idx]], [original_line(0.0), original_line(x_coord[closest_idx])], self.publisher_wall
        )
        VisualizationTools.plot_line(
            [0.0, 10.0], [offset_line(0.0), offset_line(10.0)], self.publisher_path
        )

        output = AckermannDriveStamped()
        output.header.stamp = time
        output.header.frame_id = 'base_link'
        d = ranges[closest_idx] - self.DESIRED_DISTANCE
        distance_factor = 2/3*(np.pi/2-angle_offset)*np.arctan((5*d)**3)
        angle_factor = 2/3 * 1/(1+abs(d)) * angle_offset

        look_ahead = min(msg.ranges[50:52])
        look_ahead_factor = -np.pi/(self.DESIRED_DISTANCE*look_ahead) if look_ahead < 2*self.DESIRED_DISTANCE else 0
        turn = self.SIDE*(distance_factor + look_ahead_factor) + angle_factor
        # print(look_ahead_factor)
        # print('Angle: ', angle_offset)
        # print('Angle_f: ', angle_factor)
        print('Distance: ', d)
        # print('Distance_f: ', distance_factor)
        # print('Turn: ', turn)

        output.drive.steering_angle = turn
        output.drive.speed = self.VELOCITY

        self.publisher_.publish(output)
    
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
    