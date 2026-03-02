import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped

class Straight(Node):
    def __init__(self):
        super().__init__("straight")
        self.declare_parameter("drive_topic", "/drive")
        self.DRIVE_TOPIC = self.get_parameter('drive_topic').get_parameter_value().string

        self.drive_pub = self.create_publisher(AckermannDriveStamped, self.DRIVE_TOPIC, 1)
        self.timer = self.create_timer(0.1, self.go_straight)

    def go_straight(self):
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed = 0.5
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
