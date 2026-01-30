import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class InstructionListener(Node):
    def __init__(self) -> None:
        super().__init__("instruction_listener")
        self.create_subscription(String, "/user/instruction", self.on_message, 10)
        self.get_logger().info("instruction_listener started.")

    def on_message(self, msg: String) -> None:
        self.get_logger().info(f"received: {msg.data}")


def main() -> None:
    rclpy.init()
    node = InstructionListener()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
