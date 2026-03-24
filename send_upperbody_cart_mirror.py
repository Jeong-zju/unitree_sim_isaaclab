#!/usr/bin/env python3
"""
Subscribe teleop PoseStamped topics and mirror their relative motion to the G1 arms.
"""

import argparse
import math
import os
import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from g1_upperbody_interface import ARM_JOINT_LIMITS, ARM_JOINT_ORDER, G1UpperBodyPublisher


LEFT_ARM_JOINTS = ARM_JOINT_ORDER[:7]
RIGHT_ARM_JOINTS = ARM_JOINT_ORDER[7:]


@dataclass(frozen=True)
class JointSegment:
    offset: tuple[float, float, float]
    quat: tuple[float, float, float, float]
    axis: tuple[float, float, float]


@dataclass(frozen=True)
class PoseState:
    position: np.ndarray
    rotation: np.ndarray


def _normalized(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        return vector.copy()
    return vector / norm


def _quat_to_matrix(quat: tuple[float, float, float, float]) -> np.ndarray:
    w, x, y, z = quat
    quat_array = _normalized(np.array([w, x, y, z], dtype=float))
    w, x, y, z = quat_array
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _axis_angle_to_matrix(axis: tuple[float, float, float], angle: float) -> np.ndarray:
    axis_array = _normalized(np.array(axis, dtype=float))
    x, y, z = axis_array
    cosine = math.cos(angle)
    sine = math.sin(angle)
    one_minus_cosine = 1.0 - cosine
    return np.array(
        [
            [
                cosine + x * x * one_minus_cosine,
                x * y * one_minus_cosine - z * sine,
                x * z * one_minus_cosine + y * sine,
            ],
            [
                y * x * one_minus_cosine + z * sine,
                cosine + y * y * one_minus_cosine,
                y * z * one_minus_cosine - x * sine,
            ],
            [
                z * x * one_minus_cosine - y * sine,
                z * y * one_minus_cosine + x * sine,
                cosine + z * z * one_minus_cosine,
            ],
        ],
        dtype=float,
    )


def _rotation_to_rpy(rotation: np.ndarray) -> np.ndarray:
    sy = math.sqrt(rotation[0, 0] * rotation[0, 0] + rotation[1, 0] * rotation[1, 0])
    singular = sy < 1e-8
    if not singular:
        roll = math.atan2(rotation[2, 1], rotation[2, 2])
        pitch = math.atan2(-rotation[2, 0], sy)
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = math.atan2(-rotation[1, 2], rotation[1, 1])
        pitch = math.atan2(-rotation[2, 0], sy)
        yaw = 0.0
    return np.array([roll, pitch, yaw], dtype=float)


def _rotation_matrix_to_rotvec(rotation: np.ndarray) -> np.ndarray:
    trace_value = float(np.trace(rotation))
    cosine = max(-1.0, min(1.0, 0.5 * (trace_value - 1.0)))
    angle = math.acos(cosine)
    if angle < 1e-9:
        return np.zeros(3, dtype=float)

    if abs(math.pi - angle) < 1e-4:
        diagonal = np.diag(rotation)
        axis = np.sqrt(np.maximum((diagonal + 1.0) * 0.5, 0.0))
        if axis[0] > 1e-4:
            axis[1] = math.copysign(axis[1], rotation[0, 1] + rotation[1, 0])
            axis[2] = math.copysign(axis[2], rotation[0, 2] + rotation[2, 0])
        elif axis[1] > 1e-4:
            axis[2] = math.copysign(axis[2], rotation[1, 2] + rotation[2, 1])
        axis = _normalized(axis)
        return axis * angle

    scale = angle / (2.0 * math.sin(angle))
    return scale * np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ],
        dtype=float,
    )


def _rotvec_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-9:
        return np.eye(3, dtype=float)
    axis = rotvec / angle
    return _axis_angle_to_matrix(tuple(axis.tolist()), angle)


def _rotation_distance(rotation_a: np.ndarray, rotation_b: np.ndarray) -> float:
    return float(np.linalg.norm(_rotation_matrix_to_rotvec(rotation_a.T @ rotation_b)))


def _format_xyz(position: np.ndarray) -> str:
    return f"({position[0]:+.3f}, {position[1]:+.3f}, {position[2]:+.3f})"


class ArmKinematics:
    def __init__(
        self,
        joint_names: list[str],
        segments: list[JointSegment],
        tip_offset: tuple[float, float, float],
    ):
        if len(joint_names) != len(segments):
            raise ValueError("joint_names and segments must have the same length")
        self.joint_names = tuple(joint_names)
        self.segments = tuple(segments)
        self.tip_offset = np.array(tip_offset, dtype=float)
        self.lower_limits = np.array([ARM_JOINT_LIMITS[name][0] for name in joint_names], dtype=float)
        self.upper_limits = np.array([ARM_JOINT_LIMITS[name][1] for name in joint_names], dtype=float)

    def clamp(self, joint_positions: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(joint_positions, dtype=float), self.lower_limits, self.upper_limits)

    def forward(self, joint_positions: np.ndarray):
        joint_positions = self.clamp(joint_positions)
        position = np.zeros(3, dtype=float)
        rotation = np.eye(3, dtype=float)
        joint_origins: list[np.ndarray] = []
        joint_axes: list[np.ndarray] = []

        for segment, joint_value in zip(self.segments, joint_positions):
            position = position + rotation @ np.array(segment.offset, dtype=float)
            rotation = rotation @ _quat_to_matrix(segment.quat)
            joint_origins.append(position.copy())
            joint_axes.append(rotation @ np.array(segment.axis, dtype=float))
            rotation = rotation @ _axis_angle_to_matrix(segment.axis, float(joint_value))

        ee_position = position + rotation @ self.tip_offset
        return ee_position, rotation, joint_origins, joint_axes

    def jacobian(self, joint_positions: np.ndarray) -> np.ndarray:
        ee_position, _, joint_origins, joint_axes = self.forward(joint_positions)
        jacobian = np.zeros((6, len(self.joint_names)), dtype=float)
        for index, (origin, axis) in enumerate(zip(joint_origins, joint_axes)):
            jacobian[:3, index] = np.cross(axis, ee_position - origin)
            jacobian[3:, index] = axis
        return jacobian


class DifferentialIKArm:
    def __init__(
        self,
        name: str,
        kinematics: ArmKinematics,
        rest_positions: np.ndarray | None = None,
        damping: float = 0.08,
        nullspace_gain: float = 0.12,
        max_joint_speed: float = 1.5,
    ):
        self.name = name
        self.kinematics = kinematics
        base_rest = np.zeros(len(self.kinematics.joint_names), dtype=float) if rest_positions is None else rest_positions
        self.rest_positions = self.kinematics.clamp(np.asarray(base_rest, dtype=float))
        self.joint_positions = self.rest_positions.copy()
        self.damping = float(damping)
        self.nullspace_gain = float(nullspace_gain)
        self.max_joint_speed = float(max_joint_speed)

    def reset(self):
        self.joint_positions = self.rest_positions.copy()

    def get_pose(self) -> PoseState:
        position, rotation = self.kinematics.forward(self.joint_positions)[:2]
        return PoseState(position.copy(), rotation.copy())

    def step(self, twist: np.ndarray, dt: float):
        if dt <= 0.0:
            return

        twist = np.asarray(twist, dtype=float)
        if np.linalg.norm(twist) < 1e-9:
            return

        jacobian = self.kinematics.jacobian(self.joint_positions)
        damping_squared = self.damping * self.damping
        damped_inverse = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + damping_squared * np.eye(6, dtype=float),
            np.eye(6, dtype=float),
        )
        joint_velocity = damped_inverse @ twist
        if self.nullspace_gain > 0.0:
            joint_velocity += self.nullspace_gain * (
                np.eye(len(self.joint_positions), dtype=float) - damped_inverse @ jacobian
            ) @ (self.rest_positions - self.joint_positions)
        joint_velocity = np.clip(joint_velocity, -self.max_joint_speed, self.max_joint_speed)
        self.joint_positions = self.kinematics.clamp(self.joint_positions + joint_velocity * dt)


def build_left_arm_kinematics() -> ArmKinematics:
    return ArmKinematics(
        joint_names=list(LEFT_ARM_JOINTS),
        segments=[
            JointSegment((0.0039563, 0.10022, 0.24778), (0.990264, 0.139201, 1.38722e-05, -9.86868e-05), (0.0, 1.0, 0.0)),
            JointSegment((0.0, 0.038, -0.013831), (0.990268, -0.139172, 0.0, 0.0), (1.0, 0.0, 0.0)),
            JointSegment((0.0, 0.00624, -0.1032), (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
            JointSegment((0.015783, 0.0, -0.080518), (1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            JointSegment((0.1, 0.00188791, -0.01), (1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
            JointSegment((0.038, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            JointSegment((0.046, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        ],
        tip_offset=(0.0415, 0.003, 0.0),
    )


def build_right_arm_kinematics() -> ArmKinematics:
    return ArmKinematics(
        joint_names=list(RIGHT_ARM_JOINTS),
        segments=[
            JointSegment((0.0039563, -0.10021, 0.24778), (0.990264, -0.139201, 1.38722e-05, 9.86868e-05), (0.0, 1.0, 0.0)),
            JointSegment((0.0, -0.038, -0.013831), (0.990268, 0.139172, 0.0, 0.0), (1.0, 0.0, 0.0)),
            JointSegment((0.0, -0.00624, -0.1032), (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
            JointSegment((0.015783, 0.0, -0.080518), (1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            JointSegment((0.1, -0.00188791, -0.01), (1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
            JointSegment((0.038, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            JointSegment((0.046, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        ],
        tip_offset=(0.0415, -0.003, 0.0),
    )


class TeleopPoseTracker:
    def __init__(
        self,
        name: str,
        stable_frames: int,
        position_tolerance: float,
        angle_tolerance_deg: float,
    ):
        self.name = name
        self.history = deque(maxlen=max(stable_frames, 1))
        self.position_tolerance = float(position_tolerance)
        self.angle_tolerance = math.radians(float(angle_tolerance_deg))
        self.current_pose: PoseState | None = None
        self.home_pose: PoseState | None = None

    def update(self, pose: PoseState) -> bool:
        self.current_pose = PoseState(pose.position.copy(), pose.rotation.copy())
        self.history.append(self.current_pose)
        return self._try_lock_home()

    def _try_lock_home(self) -> bool:
        if self.home_pose is not None or len(self.history) < self.history.maxlen:
            return False

        positions = np.stack([sample.position for sample in self.history], axis=0)
        position_mean = positions.mean(axis=0)
        max_position_error = max(float(np.linalg.norm(sample.position - position_mean)) for sample in self.history)

        reference_rotation = self.history[-1].rotation
        max_angle_error = max(_rotation_distance(reference_rotation, sample.rotation) for sample in self.history)

        if max_position_error > self.position_tolerance or max_angle_error > self.angle_tolerance:
            return False

        reference_pose = self.history[-1]
        self.home_pose = PoseState(reference_pose.position.copy(), reference_pose.rotation.copy())
        return True


class TeleopMirrorBridge:
    def __init__(
        self,
        publish_hz: float,
        damping: float,
        nullspace_gain: float,
        max_joint_speed: float,
        position_gain: float,
        orientation_gain: float,
        max_linear_speed: float,
        max_angular_speed: float,
        translation_scale: float,
        rotation_scale: float,
        stable_frames: int,
        stable_position_tol: float,
        stable_angle_tol_deg: float,
        network_interface: str | None = None,
    ):
        self.publish_interval = 1.0 / max(publish_hz, 1e-6)
        self.position_gain = float(position_gain)
        self.orientation_gain = float(orientation_gain)
        self.max_linear_speed = float(max_linear_speed)
        self.max_angular_speed = float(max_angular_speed)
        self.translation_scale = float(translation_scale)
        self.rotation_scale = float(rotation_scale)

        self.lock = threading.Lock()
        self.running = False
        self.thread = None
        self.last_status_print_ts = 0.0
        self.both_homes_announced = False

        self.publisher = G1UpperBodyPublisher(network_interface=network_interface)
        self.arms = {
            "left": DifferentialIKArm(
                name="left",
                kinematics=build_left_arm_kinematics(),
                damping=damping,
                nullspace_gain=nullspace_gain,
                max_joint_speed=max_joint_speed,
            ),
            "right": DifferentialIKArm(
                name="right",
                kinematics=build_right_arm_kinematics(),
                damping=damping,
                nullspace_gain=nullspace_gain,
                max_joint_speed=max_joint_speed,
            ),
        }
        self.robot_home_pose = {arm_name: arm.get_pose() for arm_name, arm in self.arms.items()}
        self.trackers = {
            "left": TeleopPoseTracker("left", stable_frames, stable_position_tol, stable_angle_tol_deg),
            "right": TeleopPoseTracker("right", stable_frames, stable_position_tol, stable_angle_tol_deg),
        }

        print("=" * 72)
        print("G1 teleop Cartesian mirror bridge")
        print("Subscribe PoseStamped topics for left/right arm end poses.")
        print("Wait for stable startup frames, lock them as teleop home poses,")
        print("then apply relative teleop deltas on top of the G1 home poses.")
        print("=" * 72)
        if network_interface:
            print(f"[DDS] interface={network_interface}")
        else:
            print("[DDS] interface=auto (pass --network_interface if DDS messages do not arrive)")
        for arm_name, pose in self.robot_home_pose.items():
            rpy_deg = np.degrees(_rotation_to_rpy(pose.rotation))
            print(
                f"[ROBOT_HOME] {arm_name} xyz={_format_xyz(pose.position)} "
                f"rpy_deg=({rpy_deg[0]:+.1f}, {rpy_deg[1]:+.1f}, {rpy_deg[2]:+.1f})"
            )

    def _compose_positions_locked(self) -> list[float]:
        return self.arms["left"].joint_positions.tolist() + self.arms["right"].joint_positions.tolist()

    def _announce_home_locked(self, arm_name: str):
        tracker = self.trackers[arm_name]
        if tracker.home_pose is None:
            return
        rpy_deg = np.degrees(_rotation_to_rpy(tracker.home_pose.rotation))
        print(
            f"[HOME] teleop {arm_name} stabilized "
            f"xyz={_format_xyz(tracker.home_pose.position)} "
            f"rpy_deg=({rpy_deg[0]:+.1f}, {rpy_deg[1]:+.1f}, {rpy_deg[2]:+.1f})"
        )
        if not self.both_homes_announced and all(
            tracker_item.home_pose is not None for tracker_item in self.trackers.values()
        ):
            self.both_homes_announced = True
            print("[HOME] both teleop home poses are locked, mirroring is active")

    def update_teleop_pose(
        self,
        arm_name: str,
        position: tuple[float, float, float],
        quaternion_xyzw: tuple[float, float, float, float],
    ):
        rotation = _quat_to_matrix(
            (
                float(quaternion_xyzw[3]),
                float(quaternion_xyzw[0]),
                float(quaternion_xyzw[1]),
                float(quaternion_xyzw[2]),
            )
        )
        pose = PoseState(np.array(position, dtype=float), rotation)

        stabilized = False
        with self.lock:
            stabilized = self.trackers[arm_name].update(pose)
        if stabilized:
            self._announce_home_locked(arm_name)

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._publish_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        self.reset_and_publish()

    def reset_and_publish(self):
        with self.lock:
            for arm in self.arms.values():
                arm.reset()
            positions = self._compose_positions_locked()
        self.publisher.publish_positions(positions)

    def _compute_target_pose_locked(self, arm_name: str) -> PoseState:
        tracker = self.trackers[arm_name]
        robot_home = self.robot_home_pose[arm_name]
        if tracker.home_pose is None or tracker.current_pose is None:
            return PoseState(robot_home.position.copy(), robot_home.rotation.copy())

        teleop_home = tracker.home_pose
        teleop_current = tracker.current_pose

        delta_position_local = teleop_home.rotation.T @ (teleop_current.position - teleop_home.position)
        delta_rotation = teleop_home.rotation.T @ teleop_current.rotation
        delta_rotvec = _rotation_matrix_to_rotvec(delta_rotation)

        target_position = robot_home.position + robot_home.rotation @ (self.translation_scale * delta_position_local)
        target_rotation = robot_home.rotation @ _rotvec_to_matrix(self.rotation_scale * delta_rotvec)
        return PoseState(target_position, target_rotation)

    def _build_twist(self, current_pose: PoseState, target_pose: PoseState) -> np.ndarray:
        position_error = target_pose.position - current_pose.position
        angular_error = _rotation_matrix_to_rotvec(target_pose.rotation @ current_pose.rotation.T)

        linear_cmd = self.position_gain * position_error
        angular_cmd = self.orientation_gain * angular_error

        linear_norm = float(np.linalg.norm(linear_cmd))
        if linear_norm > self.max_linear_speed > 0.0:
            linear_cmd *= self.max_linear_speed / linear_norm

        angular_norm = float(np.linalg.norm(angular_cmd))
        if angular_norm > self.max_angular_speed > 0.0:
            angular_cmd *= self.max_angular_speed / angular_norm

        return np.concatenate([linear_cmd, angular_cmd])

    def _publish_loop(self):
        last_tick = time.monotonic()
        while self.running:
            loop_start = time.monotonic()
            dt = min(loop_start - last_tick, 0.1)
            last_tick = loop_start

            with self.lock:
                target_pose = {
                    arm_name: self._compute_target_pose_locked(arm_name) for arm_name in ("left", "right")
                }

                for arm_name in ("left", "right"):
                    current_pose = self.arms[arm_name].get_pose()
                    twist = self._build_twist(current_pose, target_pose[arm_name])
                    self.arms[arm_name].step(twist, dt)

                positions = self._compose_positions_locked()

                now = time.monotonic()
                if now - self.last_status_print_ts > 1.0:
                    left_home_ready = self.trackers["left"].home_pose is not None
                    right_home_ready = self.trackers["right"].home_pose is not None
                    print(
                        "[STATUS] "
                        f"left_home={'yes' if left_home_ready else 'no'} "
                        f"right_home={'yes' if right_home_ready else 'no'} "
                        f"left_target={_format_xyz(target_pose['left'].position)} "
                        f"right_target={_format_xyz(target_pose['right'].position)}"
                    )
                    self.last_status_print_ts = now

            self.publisher.publish_positions(positions)

            elapsed = time.monotonic() - loop_start
            if elapsed < self.publish_interval:
                time.sleep(self.publish_interval - elapsed)


def run_ros2(bridge: TeleopMirrorBridge, left_topic: str, right_topic: str):
    try:
        import rclpy
        from geometry_msgs.msg import PoseStamped
        from rclpy.node import Node
    except ImportError as exc:
        raise RuntimeError(
            "ROS 2 dependencies are missing. Please install rclpy and geometry_msgs."
        ) from exc

    class TeleopPoseNode(Node):
        def __init__(self):
            super().__init__("teleop_pose_to_g1_bridge")
            self.create_subscription(PoseStamped, left_topic, self._left_callback, 10)
            self.create_subscription(PoseStamped, right_topic, self._right_callback, 10)

        def _left_callback(self, msg: PoseStamped):
            bridge.update_teleop_pose(
                "left",
                (msg.pose.position.x, msg.pose.position.y, msg.pose.position.z),
                (
                    msg.pose.orientation.x,
                    msg.pose.orientation.y,
                    msg.pose.orientation.z,
                    msg.pose.orientation.w,
                ),
            )

        def _right_callback(self, msg: PoseStamped):
            bridge.update_teleop_pose(
                "right",
                (msg.pose.position.x, msg.pose.position.y, msg.pose.position.z),
                (
                    msg.pose.orientation.x,
                    msg.pose.orientation.y,
                    msg.pose.orientation.z,
                    msg.pose.orientation.w,
                ),
            )

    rclpy.init()
    node = TeleopPoseNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


def run_ros1(bridge: TeleopMirrorBridge, left_topic: str, right_topic: str):
    try:
        import rospy
        from geometry_msgs.msg import PoseStamped
    except ImportError as exc:
        raise RuntimeError(
            "ROS 1 dependencies are missing. Please install rospy and geometry_msgs."
        ) from exc

    def left_callback(msg: PoseStamped):
        bridge.update_teleop_pose(
            "left",
            (msg.pose.position.x, msg.pose.position.y, msg.pose.position.z),
            (
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
                msg.pose.orientation.w,
            ),
        )

    def right_callback(msg: PoseStamped):
        bridge.update_teleop_pose(
            "right",
            (msg.pose.position.x, msg.pose.position.y, msg.pose.position.z),
            (
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
                msg.pose.orientation.w,
            ),
        )

    rospy.init_node("teleop_pose_to_g1_bridge", anonymous=True)
    rospy.Subscriber(left_topic, PoseStamped, left_callback, queue_size=10)
    rospy.Subscriber(right_topic, PoseStamped, right_callback, queue_size=10)
    rospy.spin()


def main():
    parser = argparse.ArgumentParser(description="Mirror teleop arm PoseStamped deltas to the G1 upper body")
    parser.add_argument("--left_topic", type=str, default="/teleop/arm_left/end_pose", help="ROS topic for the left teleop arm pose")
    parser.add_argument("--right_topic", type=str, default="/teleop/arm_right/end_pose", help="ROS topic for the right teleop arm pose")
    parser.add_argument(
        "--ros_version",
        type=str,
        default="auto",
        choices=["auto", "1", "2"],
        help="ROS version to use for subscription",
    )
    parser.add_argument("--publish_hz", type=float, default=50.0, help="DDS publish frequency")
    parser.add_argument("--network_interface", type=str, default=None, help="DDS network interface, e.g. enp3s0")
    parser.add_argument("--stable_frames", type=int, default=30, help="Number of startup frames used for home-pose stabilization")
    parser.add_argument("--stable_position_tol", type=float, default=0.01, help="Maximum position deviation for home-pose locking in meters")
    parser.add_argument("--stable_angle_tol_deg", type=float, default=5.0, help="Maximum orientation deviation for home-pose locking in degrees")
    parser.add_argument("--translation_scale", type=float, default=1.0, help="Scale applied to teleop translation deltas")
    parser.add_argument("--rotation_scale", type=float, default=1.0, help="Scale applied to teleop rotation deltas")
    parser.add_argument("--position_gain", type=float, default=4.0, help="Position tracking gain")
    parser.add_argument("--orientation_gain", type=float, default=5.0, help="Orientation tracking gain")
    parser.add_argument("--max_linear_speed", type=float, default=0.35, help="Maximum Cartesian tracking linear speed in m/s")
    parser.add_argument("--max_angular_speed", type=float, default=2.0, help="Maximum Cartesian tracking angular speed in rad/s")
    parser.add_argument("--damping", type=float, default=0.08, help="Damped least-squares factor")
    parser.add_argument("--nullspace_gain", type=float, default=0.12, help="Gain for biasing the arms back to the default posture")
    parser.add_argument("--joint_speed_limit", type=float, default=1.5, help="Per-joint IK speed limit in rad/s")
    args = parser.parse_args()

    network_interface = args.network_interface or os.getenv("UNITREE_DDS_INTERFACE")

    bridge = TeleopMirrorBridge(
        publish_hz=args.publish_hz,
        damping=args.damping,
        nullspace_gain=args.nullspace_gain,
        max_joint_speed=args.joint_speed_limit,
        position_gain=args.position_gain,
        orientation_gain=args.orientation_gain,
        max_linear_speed=args.max_linear_speed,
        max_angular_speed=args.max_angular_speed,
        translation_scale=args.translation_scale,
        rotation_scale=args.rotation_scale,
        stable_frames=args.stable_frames,
        stable_position_tol=args.stable_position_tol,
        stable_angle_tol_deg=args.stable_angle_tol_deg,
        network_interface=network_interface,
    )
    bridge.start()

    try:
        if args.ros_version == "2":
            run_ros2(bridge, args.left_topic, args.right_topic)
        elif args.ros_version == "1":
            run_ros1(bridge, args.left_topic, args.right_topic)
        else:
            try:
                run_ros2(bridge, args.left_topic, args.right_topic)
            except RuntimeError as ros2_error:
                print(f"ROS 2 unavailable: {ros2_error}")
                print("Falling back to ROS 1...")
                run_ros1(bridge, args.left_topic, args.right_topic)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except RuntimeError as exc:
        print(f"Bridge startup failed: {exc}")
    finally:
        bridge.stop()


if __name__ == "__main__":
    main()
