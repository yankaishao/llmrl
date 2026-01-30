import sys

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class InstructionSource(Node):
    def __init__(self) -> None:
        super().__init__("instruction_source")
        self.publisher_ = self.create_publisher(String, "/user/instruction", 10)
        self.get_logger().info("instruction_source started; type a line and press Enter.")

    def publish_line(self, line: str) -> None:
        msg = String()
        msg.data = line
        self.publisher_.publish(msg)
        self.get_logger().info(f"published: {line}")


def main() -> None:
    rclpy.init()
    node = InstructionSource()
    try:
        while rclpy.ok():
            line = sys.stdin.readline()
            if line == "":
                break
            line = line.rstrip("\n")
            if not line:
                continue
            node.publish_line(line)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
