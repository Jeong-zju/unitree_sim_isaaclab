#!/usr/bin/env python3
"""
Subscribe /cmd_vel (geometry_msgs/Twist) and bridge it to rt/run_command/cmd.
"""

import argparse
import shutil
import subprocess
import threading
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_


class CmdVelBridge:
    def __init__(
        self,
        publisher: ChannelPublisher,
        publish_hz: float,
        default_height: float,
        cmd_timeout: float,
        x_range: tuple[float, float],
        y_range: tuple[float, float],
        yaw_range: tuple[float, float],
    ):
        self.publisher = publisher
        self.publish_interval = 1.0 / max(publish_hz, 1e-6)
        self.default_height = default_height
        self.cmd_timeout = cmd_timeout
        self.x_range = x_range
        self.y_range = y_range
        self.yaw_range = yaw_range

        self._lock = threading.Lock()
        self._running = False
        self._publisher_thread = None
        self._last_cmd_time = 0.0
        self._last_published = None
        self._cmd = {
            "x_vel": 0.0,
            "y_vel": 0.0,
            "yaw_vel": 0.0,
        }

    def _clamp(self, value: float, limits: tuple[float, float]) -> float:
        return max(limits[0], min(limits[1], float(value)))

    def update_from_twist(self, linear_x: float, linear_y: float, angular_z: float):
        with self._lock:
            self._cmd["x_vel"] = self._clamp(linear_x, self.x_range)
            self._cmd["y_vel"] = self._clamp(linear_y, self.y_range)
            self._cmd["yaw_vel"] = self._clamp(angular_z, self.yaw_range)
            self._last_cmd_time = time.time()

    def _get_command_list(self) -> list[float]:
        with self._lock:
            command = dict(self._cmd)
            last_cmd_time = self._last_cmd_time

        if self.cmd_timeout > 0.0 and (time.time() - last_cmd_time) > self.cmd_timeout:
            command["x_vel"] = 0.0
            command["y_vel"] = 0.0
            command["yaw_vel"] = 0.0

        return [
            round(float(command["x_vel"]), 3),
            round(float(command["y_vel"]), 3),
            round(float(command["yaw_vel"]), 3),
            round(float(self.default_height), 3),
        ]

    def start(self):
        if self._running:
            return
        self._running = True
        self._publisher_thread = threading.Thread(target=self._publish_loop, daemon=True)
        self._publisher_thread.start()

    def stop(self):
        self._running = False
        if self._publisher_thread is not None:
            self._publisher_thread.join(timeout=1.0)
        self.publisher.Write(String_(data=str([0.0, 0.0, 0.0, float(self.default_height)])))

    def _publish_loop(self):
        while self._running:
            command_list = self._get_command_list()
            if command_list != self._last_published:
                print(f"cmd_vel -> DDS: {command_list}")
                self._last_published = command_list
            self.publisher.Write(String_(data=str(command_list)))
            time.sleep(self.publish_interval)


def run_ros2(bridge: CmdVelBridge, topic_name: str):
    try:
        import rclpy
        from geometry_msgs.msg import Twist
        from rclpy.node import Node
    except ImportError as exc:
        raise RuntimeError(
            "ROS 2 dependencies are missing. Please install rclpy and geometry_msgs."
        ) from exc

    class CmdVelNode(Node):
        def __init__(self):
            super().__init__("cmd_vel_to_unitree_dds_bridge")
            self.create_subscription(Twist, topic_name, self._callback, 10)

        def _callback(self, msg: Twist):
            bridge.update_from_twist(msg.linear.x, msg.linear.y, msg.angular.z)

    rclpy.init()
    node = CmdVelNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


def run_ros1_python(bridge: CmdVelBridge, topic_name: str):
    try:
        import rospy
        from geometry_msgs.msg import Twist
    except ImportError as exc:
        raise RuntimeError(
            "ROS 1 dependencies are missing. Please install rospy and geometry_msgs."
        ) from exc

    def callback(msg: Twist):
        bridge.update_from_twist(msg.linear.x, msg.linear.y, msg.angular.z)

    rospy.init_node("cmd_vel_to_unitree_dds_bridge", anonymous=True)
    rospy.Subscriber(topic_name, Twist, callback, queue_size=10)
    rospy.spin()


def run_ros1_rostopic(bridge: CmdVelBridge, topic_name: str):
    rostopic_bin = shutil.which("rostopic")
    if rostopic_bin is None:
        raise RuntimeError(
            "ROS 1 Python packages are unavailable and 'rostopic' is not in PATH. "
            "Please source your ROS1 setup.bash before running this script."
        )

    process = subprocess.Popen(
        [rostopic_bin, "echo", "-p", topic_name],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    header = None
    field_indices = None
    bootstrap_logs = []

    try:
        assert process.stdout is not None
        for raw_line in process.stdout:
            line = raw_line.strip()
            if not line:
                continue

            if header is None:
                if line.startswith("%time,"):
                    header = [field.strip() for field in line.split(",")]
                    try:
                        field_indices = {
                            "linear_x": header.index("field.linear.x"),
                            "linear_y": header.index("field.linear.y"),
                            "angular_z": header.index("field.angular.z"),
                        }
                    except ValueError as exc:
                        raise RuntimeError(
                            f"Unexpected rostopic CSV header for {topic_name}: {line}"
                        ) from exc
                    print(f"Subscribed to ROS1 topic via rostopic: {topic_name}")
                    continue

                bootstrap_logs.append(line)
                if len(bootstrap_logs) > 10:
                    bootstrap_logs.pop(0)
                continue

            values = [field.strip() for field in line.split(",")]
            if field_indices is None or len(values) <= max(field_indices.values()):
                continue

            try:
                bridge.update_from_twist(
                    float(values[field_indices["linear_x"]]),
                    float(values[field_indices["linear_y"]]),
                    float(values[field_indices["angular_z"]]),
                )
            except ValueError:
                continue

        return_code = process.wait(timeout=1.0)
        if header is None:
            logs = "\n".join(bootstrap_logs) if bootstrap_logs else "no output from rostopic"
            raise RuntimeError(
                f"Failed to subscribe to ROS1 topic {topic_name} via rostopic. Output:\n{logs}"
            )
        raise RuntimeError(f"rostopic exited unexpectedly with code {return_code}")
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.kill()


def run_ros1(bridge: CmdVelBridge, topic_name: str):
    try:
        run_ros1_python(bridge, topic_name)
    except RuntimeError as rospy_error:
        print(f"rospy unavailable: {rospy_error}")
        print("Falling back to 'rostopic echo -p' subscriber...")
        run_ros1_rostopic(bridge, topic_name)


def main():
    parser = argparse.ArgumentParser(description="Bridge /cmd_vel to Unitree DDS base command")
    parser.add_argument("--cmd_vel_topic", type=str, default="/cmd_vel", help="ROS Twist topic name")
    parser.add_argument(
        "--ros_version",
        type=str,
        default="auto",
        choices=["auto", "1", "2"],
        help="ROS version to use for subscription",
    )
    parser.add_argument("--network_interface", type=str, default=None, help="DDS network interface, e.g. enp3s0")
    parser.add_argument("--publish_hz", type=float, default=50.0, help="DDS publish frequency")
    parser.add_argument("--cmd_timeout", type=float, default=0.5, help="Seconds before zeroing stale cmd_vel")
    parser.add_argument("--default_height", type=float, default=0.8, help="Fixed height command")
    parser.add_argument("--x_min", type=float, default=-0.6, help="Minimum x velocity")
    parser.add_argument("--x_max", type=float, default=1.0, help="Maximum x velocity")
    parser.add_argument("--y_min", type=float, default=-0.5, help="Minimum y velocity")
    parser.add_argument("--y_max", type=float, default=0.5, help="Maximum y velocity")
    parser.add_argument("--yaw_min", type=float, default=-1.57, help="Minimum yaw velocity")
    parser.add_argument("--yaw_max", type=float, default=1.57, help="Maximum yaw velocity")
    args = parser.parse_args()

    ChannelFactoryInitialize(1, args.network_interface)
    publisher = ChannelPublisher("rt/run_command/cmd", String_)
    publisher.Init()

    bridge = CmdVelBridge(
        publisher=publisher,
        publish_hz=args.publish_hz,
        default_height=args.default_height,
        cmd_timeout=args.cmd_timeout,
        x_range=(args.x_min, args.x_max),
        y_range=(args.y_min, args.y_max),
        yaw_range=(args.yaw_min, args.yaw_max),
    )
    bridge.start()

    try:
        if args.ros_version == "2":
            run_ros2(bridge, args.cmd_vel_topic)
        elif args.ros_version == "1":
            run_ros1(bridge, args.cmd_vel_topic)
        else:
            try:
                run_ros2(bridge, args.cmd_vel_topic)
            except RuntimeError as ros2_error:
                print(f"ROS 2 unavailable: {ros2_error}")
                print("Falling back to ROS 1...")
                run_ros1(bridge, args.cmd_vel_topic)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except RuntimeError as exc:
        print(f"Bridge startup failed: {exc}")
    finally:
        bridge.stop()


if __name__ == "__main__":
    main()
