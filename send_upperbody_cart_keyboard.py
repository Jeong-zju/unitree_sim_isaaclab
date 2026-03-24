#!/usr/bin/env python3

import argparse
import math
import os
import select
import sys
import termios
import threading
import time
import tty
from dataclasses import dataclass

import numpy as np

from g1_upperbody_interface import ARM_JOINT_LIMITS, ARM_JOINT_ORDER, G1UpperBodyPublisher


LEFT_ARM_JOINTS = ARM_JOINT_ORDER[:7]
RIGHT_ARM_JOINTS = ARM_JOINT_ORDER[7:]
ACTIVE_KEYS = tuple("wsadrfijkluo")
MODE_LABELS = {
    "left": "left arm",
    "right": "right arm",
    "dual": "dual-arm mirror",
}


@dataclass(frozen=True)
class JointSegment:
    offset: tuple[float, float, float]
    quat: tuple[float, float, float, float]
    axis: tuple[float, float, float]


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

    def get_pose(self):
        return self.kinematics.forward(self.joint_positions)[:2]

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


class UpperBodyCartesianKeyboardController:
    def __init__(
        self,
        keyboard_module,
        linear_speed: float,
        angular_speed: float,
        damping: float,
        nullspace_gain: float,
        max_joint_speed: float,
        input_mode: str = "auto",
        network_interface: str | None = None,
    ):
        self.linear_speed = float(linear_speed)
        self.angular_speed = float(angular_speed)
        self.speed_scale = 1.0
        self.running = True
        self.mode = "left"
        self.state_lock = threading.Lock()
        self.key_states = {key: False for key in ACTIVE_KEYS}
        self.terminal_key_deadlines = {key: 0.0 for key in ACTIVE_KEYS}
        self.terminal_key_timeout = 0.18
        self.keyboard_module = keyboard_module
        self.global_listener = None
        self.terminal_thread = None
        self.terminal_reader_running = False
        self.terminal_fd = None
        self.terminal_old_settings = None
        self.last_motion_report_ts = 0.0
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
        self._start_input_backend(input_mode)
        self._print_help()
        if network_interface:
            print(f"[DDS] interface={network_interface}")
        else:
            print("[DDS] interface=auto (pass --network_interface if DDS messages do not arrive)")
        self._print_state()

    def _start_input_backend(self, input_mode: str):
        resolved_mode = input_mode
        if input_mode == "auto":
            resolved_mode = "terminal" if sys.stdin.isatty() else "global"

        if resolved_mode == "terminal":
            if not sys.stdin.isatty():
                raise RuntimeError("terminal input mode requires a TTY")
            self._start_terminal_input()
            print("[INPUT] terminal mode active, keep focus on this terminal")
            return

        if resolved_mode == "global":
            if self.keyboard_module is None:
                raise RuntimeError("global input mode requires pynput")
            self._start_global_input()
            print("[INPUT] global pynput mode active")
            return

        raise ValueError(f"unsupported input mode: {input_mode}")

    def _start_global_input(self):
        self.global_listener = self.keyboard_module.Listener(on_press=self._on_press, on_release=self._on_release)
        self.global_listener.start()

    def _start_terminal_input(self):
        self.terminal_fd = sys.stdin.fileno()
        self.terminal_old_settings = termios.tcgetattr(self.terminal_fd)
        tty.setcbreak(self.terminal_fd)
        self.terminal_reader_running = True
        self.terminal_thread = threading.Thread(target=self._terminal_input_loop, daemon=True)
        self.terminal_thread.start()

    def _terminal_input_loop(self):
        try:
            while self.running and self.terminal_reader_running:
                ready, _, _ = select.select([sys.stdin], [], [], 0.05)
                if not ready:
                    continue
                key_char = sys.stdin.read(1)
                if not key_char:
                    continue
                if key_char == "\x03":
                    self.running = False
                    break
                self._handle_key_char(key_char.lower(), terminal_input=True)
        finally:
            self._restore_terminal()

    def _restore_terminal(self):
        if self.terminal_fd is None or self.terminal_old_settings is None:
            return
        try:
            termios.tcsetattr(self.terminal_fd, termios.TCSADRAIN, self.terminal_old_settings)
        except termios.error:
            pass
        finally:
            self.terminal_fd = None
            self.terminal_old_settings = None

    def _print_help(self):
        print("=" * 72)
        print("G1 dual-arm Cartesian keyboard control")
        print("Mode: 1 left arm, 2 right arm, 3 dual-arm mirror")
        print("Position  W/S: +/-X   A/D: +/-Y   R/F: +/-Z")
        print("Rotation  I/K: roll   J/L: pitch  U/O: yaw")
        print(",: slower   .: faster")
        print("Z: reset active arm(s)   X: reset both arms   P: print state   H: help   Q: exit")
        print("In dual-arm mirror mode, Y / roll / yaw commands are mirrored for the right arm.")
        print("=" * 72)

    def _mode_label(self) -> str:
        return MODE_LABELS[self.mode]

    def _current_speed_line(self) -> str:
        return (
            f"[SPEED] linear={self.linear_speed * self.speed_scale:.3f} m/s  "
            f"angular={self.angular_speed * self.speed_scale:.3f} rad/s"
        )

    def _compose_positions_locked(self) -> list[float]:
        return self.arms["left"].joint_positions.tolist() + self.arms["right"].joint_positions.tolist()

    def _active_arm_names_locked(self) -> list[str]:
        if self.mode == "left":
            return ["left"]
        if self.mode == "right":
            return ["right"]
        return ["left", "right"]

    def _reset_active_locked(self):
        for arm_name in self._active_arm_names_locked():
            self.arms[arm_name].reset()

    def _reset_all_locked(self):
        for arm in self.arms.values():
            arm.reset()

    def _build_state_lines_locked(self) -> list[str]:
        lines = [f"[MODE] {self._mode_label()}", self._current_speed_line()]
        for arm_name in ("left", "right"):
            position, rotation = self.arms[arm_name].get_pose()
            rpy_deg = np.degrees(_rotation_to_rpy(rotation))
            joint_values = ", ".join(f"{value:+.3f}" for value in self.arms[arm_name].joint_positions)
            lines.append(
                f"[{arm_name.upper()}] xyz=({position[0]:+.3f}, {position[1]:+.3f}, {position[2]:+.3f}) "
                f"rpy_deg=({rpy_deg[0]:+.1f}, {rpy_deg[1]:+.1f}, {rpy_deg[2]:+.1f})"
            )
            lines.append(f"[{arm_name.upper()}] q=[{joint_values}]")
        return lines

    def _print_state(self):
        with self.state_lock:
            lines = self._build_state_lines_locked()
        for line in lines:
            print(line)

    def _adjust_speed_scale(self, factor: float):
        with self.state_lock:
            self.speed_scale = min(4.0, max(0.25, self.speed_scale * factor))
            speed_line = self._current_speed_line()
        print(speed_line)

    def _extract_key_char(self, key):
        try:
            return key.char.lower() if hasattr(key, "char") and key.char else None
        except AttributeError:
            return None

    def _handle_key_char(self, key_char: str, terminal_input: bool = False):
        if key_char == "q":
            self.running = False
            return False
        if key_char == "h":
            self._print_help()
            return None
        if key_char == ",":
            self._adjust_speed_scale(0.5)
            return None
        if key_char == ".":
            self._adjust_speed_scale(2.0)
            return None
        if key_char == "p":
            self._print_state()
            return None

        if key_char in ACTIVE_KEYS:
            with self.state_lock:
                if terminal_input:
                    self.terminal_key_deadlines[key_char] = time.monotonic() + self.terminal_key_timeout
                else:
                    self.key_states[key_char] = True
            return None

        if key_char in {"1", "2", "3", "z", "x"}:
            with self.state_lock:
                if key_char == "1":
                    self.mode = "left"
                    mode_line = f"[MODE] {self._mode_label()}"
                elif key_char == "2":
                    self.mode = "right"
                    mode_line = f"[MODE] {self._mode_label()}"
                elif key_char == "3":
                    self.mode = "dual"
                    mode_line = f"[MODE] {self._mode_label()}"
                elif key_char == "z":
                    self._reset_active_locked()
                    active_labels = ", ".join(self._active_arm_names_locked())
                    mode_line = f"[RESET] {active_labels}"
                else:
                    self._reset_all_locked()
                    mode_line = "[RESET] left, right"
            print(mode_line)
        return None

    def _on_press(self, key):
        key_char = self._extract_key_char(key)
        if key_char is None:
            return None
        return self._handle_key_char(key_char, terminal_input=False)

    def _on_release(self, key):
        key_char = self._extract_key_char(key)
        if key_char in ACTIVE_KEYS:
            with self.state_lock:
                self.key_states[key_char] = False

    def _compute_twist_from_keys_locked(self) -> np.ndarray:
        now = time.monotonic()

        def active(key_name: str) -> float:
            return float(self.key_states[key_name] or self.terminal_key_deadlines[key_name] > now)

        linear_axis = np.array(
            [
                active("w") - active("s"),
                active("a") - active("d"),
                active("r") - active("f"),
            ],
            dtype=float,
        )
        angular_axis = np.array(
            [
                active("i") - active("k"),
                active("j") - active("l"),
                active("u") - active("o"),
            ],
            dtype=float,
        )
        linear_axis_norm = np.linalg.norm(linear_axis)
        if linear_axis_norm > 1.0:
            linear_axis /= linear_axis_norm
        angular_axis_norm = np.linalg.norm(angular_axis)
        if angular_axis_norm > 1.0:
            angular_axis /= angular_axis_norm

        linear_twist = linear_axis * self.linear_speed * self.speed_scale
        angular_twist = angular_axis * self.angular_speed * self.speed_scale
        return np.concatenate([linear_twist, angular_twist])

    def step(self, dt: float):
        with self.state_lock:
            base_twist = self._compute_twist_from_keys_locked()

            if self.mode == "left":
                left_twist = base_twist
                right_twist = np.zeros(6, dtype=float)
            elif self.mode == "right":
                left_twist = np.zeros(6, dtype=float)
                right_twist = base_twist
            else:
                left_twist = base_twist.copy()
                right_twist = base_twist.copy()
                right_twist[1] *= -1.0
                right_twist[3] *= -1.0
                right_twist[5] *= -1.0

            self.arms["left"].step(left_twist, dt)
            self.arms["right"].step(right_twist, dt)
            positions = self._compose_positions_locked()
            motion_active = bool(np.linalg.norm(base_twist) > 1e-8)
            if motion_active:
                now = time.monotonic()
                if now - self.last_motion_report_ts > 0.5:
                    left_pos, _ = self.arms["left"].get_pose()
                    right_pos, _ = self.arms["right"].get_pose()
                    print(
                        "[CMD] "
                        f"mode={self._mode_label()} "
                        f"L=({left_pos[0]:+.3f}, {left_pos[1]:+.3f}, {left_pos[2]:+.3f}) "
                        f"R=({right_pos[0]:+.3f}, {right_pos[1]:+.3f}, {right_pos[2]:+.3f})"
                    )
                    self.last_motion_report_ts = now

        self.publisher.publish_positions(positions)

    def stop(self):
        self.running = False
        self.terminal_reader_running = False
        if self.global_listener is not None:
            self.global_listener.stop()
        if self.terminal_thread is not None and self.terminal_thread.is_alive():
            self.terminal_thread.join(timeout=0.2)
        self._restore_terminal()

    def reset_and_publish(self):
        with self.state_lock:
            self._reset_all_locked()
            positions = self._compose_positions_locked()
        self.publisher.publish_positions(positions)


def main():
    parser = argparse.ArgumentParser(description="G1 dual-arm Cartesian keyboard command sender")
    parser.add_argument(
        "--linear-speed",
        type=float,
        default=0.10,
        help="Cartesian translation speed in meters per second",
    )
    parser.add_argument(
        "--angular-speed",
        type=float,
        default=0.70,
        help="Cartesian rotation speed in radians per second",
    )
    parser.add_argument(
        "--publish-hz",
        type=float,
        default=50.0,
        help="control and publish frequency",
    )
    parser.add_argument(
        "--damping",
        type=float,
        default=0.08,
        help="damped least-squares factor",
    )
    parser.add_argument(
        "--nullspace-gain",
        type=float,
        default=0.12,
        help="gain for biasing the arms back to the default posture",
    )
    parser.add_argument(
        "--joint-speed-limit",
        type=float,
        default=1.5,
        help="per-joint speed limit in rad/s for the IK output",
    )
    parser.add_argument(
        "--input-mode",
        choices=("auto", "terminal", "global"),
        default="auto",
        help="keyboard input backend",
    )
    parser.add_argument(
        "--network_interface",
        type=str,
        default=None,
        help="DDS network interface, e.g. enp3s0",
    )
    args = parser.parse_args()

    network_interface = args.network_interface or os.getenv("UNITREE_DDS_INTERFACE")

    keyboard = None
    if args.input_mode in ("auto", "global"):
        try:
            from pynput import keyboard as pynput_keyboard

            keyboard = pynput_keyboard
        except ImportError:
            if args.input_mode == "global":
                print("error: pynput library missing")
                print("please install: pip install pynput")
                return

    try:
        controller = UpperBodyCartesianKeyboardController(
            keyboard_module=keyboard,
            linear_speed=args.linear_speed,
            angular_speed=args.angular_speed,
            damping=args.damping,
            nullspace_gain=args.nullspace_gain,
            max_joint_speed=args.joint_speed_limit,
            input_mode=args.input_mode,
            network_interface=network_interface,
        )
    except RuntimeError as error:
        print(f"error: {error}")
        return

    publish_interval = 1.0 / max(args.publish_hz, 1e-6)
    last_tick = time.monotonic()

    try:
        while controller.running:
            loop_start = time.monotonic()
            dt = min(loop_start - last_tick, 0.1)
            last_tick = loop_start
            controller.step(dt)
            elapsed = time.monotonic() - loop_start
            if elapsed < publish_interval:
                time.sleep(publish_interval - elapsed)
    except KeyboardInterrupt:
        pass
    finally:
        controller.stop()
        controller.reset_and_publish()
        print("upper-body Cartesian command sender stopped")


if __name__ == "__main__":
    main()
