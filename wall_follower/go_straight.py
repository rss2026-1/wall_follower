import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped

class Straight(Node):
    def __init__(self):
        super().__init__("straight")

        self.drive_pub = self.create_publisher(AckermannDriveStamped, "/vesc/low_level/input/navigation", 1)
        self.timer = self.create_timer(0.1, self.go_straight)

    def go_straight(self):
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.drive.speed = 2.5
        drive_msg.drive.steering_angle = 0.0
        self.drive_pub.publish(drive_msg)

def main(args=None):
    rclpy.init(args=args)
    straight_node = Straight()
    rclpy.spin(straight_node)
    straight_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
