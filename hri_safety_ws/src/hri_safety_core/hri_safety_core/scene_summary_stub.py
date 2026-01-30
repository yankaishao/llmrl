import rclpy
from rclpy.node import Node
from std_msgs.msg import String

DEFAULT_SCENE_SUMMARY = (
    "objects=[cup_red_1(left), cup_red_2(right), knife_1(center,risky)], "
    "context=[child_present=false], task_state=idle"
)


class SceneSummaryStub(Node):
    def __init__(self) -> None:
        super().__init__("scene_summary_stub")
        self.publisher_ = self.create_publisher(String, "/scene/summary", 10)
        self.summary = DEFAULT_SCENE_SUMMARY
        self.get_logger().info("scene_summary_stub started; publishing at 1 Hz.")
        self.timer = self.create_timer(1.0, self.publish_summary)
        self.publish_summary()

    def publish_summary(self) -> None:
        msg = String()
        msg.data = self.summary
        self.publisher_.publish(msg)


def main() -> None:
    rclpy.init()
    node = SceneSummaryStub()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
