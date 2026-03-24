#!/usr/bin/env python3

import json
from typing import Mapping

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_


ARM_COMMAND_TOPIC = "rt/arm_command/cmd"
ARM_JOINT_ORDER = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
ARM_JOINT_LIMITS = {
    "left_shoulder_pitch_joint": (-1.5, 1.5),
    "left_shoulder_roll_joint": (-1.2, 1.2),
    "left_shoulder_yaw_joint": (-1.5, 1.5),
    "left_elbow_joint": (0.0, 1.8),
    "left_wrist_roll_joint": (-1.8, 1.8),
    "left_wrist_pitch_joint": (-1.2, 1.2),
    "left_wrist_yaw_joint": (-1.8, 1.8),
    "right_shoulder_pitch_joint": (-1.5, 1.5),
    "right_shoulder_roll_joint": (-1.2, 1.2),
    "right_shoulder_yaw_joint": (-1.5, 1.5),
    "right_elbow_joint": (0.0, 1.8),
    "right_wrist_roll_joint": (-1.8, 1.8),
    "right_wrist_pitch_joint": (-1.2, 1.2),
    "right_wrist_yaw_joint": (-1.8, 1.8),
}


def clamp_arm_target(joint_name: str, value: float) -> float:
    low, high = ARM_JOINT_LIMITS[joint_name]
    return max(low, min(high, float(value)))


class G1UpperBodyPublisher:
    def __init__(self, dds_channel: int = 1, network_interface: str = None):
        ChannelFactoryInitialize(dds_channel, network_interface)
        self.publisher = ChannelPublisher(ARM_COMMAND_TOPIC, String_)
        self.publisher.Init()
        self.targets = {joint_name: 0.0 for joint_name in ARM_JOINT_ORDER}

    def set_joint_target(self, joint_name: str, value: float):
        if joint_name not in self.targets:
            raise KeyError(f"Unknown joint name: {joint_name}")
        self.targets[joint_name] = clamp_arm_target(joint_name, value)

    def update_joint_targets(self, joint_targets: Mapping[str, float]):
        for joint_name, value in joint_targets.items():
            self.set_joint_target(joint_name, value)

    def set_positions(self, positions: list[float]):
        if len(positions) < len(ARM_JOINT_ORDER):
            raise ValueError(f"Expected at least {len(ARM_JOINT_ORDER)} positions")
        for joint_name, value in zip(ARM_JOINT_ORDER, positions):
            self.set_joint_target(joint_name, value)

    def reset_targets(self):
        for joint_name in ARM_JOINT_ORDER:
            self.targets[joint_name] = 0.0

    def get_positions(self) -> list[float]:
        return [self.targets[joint_name] for joint_name in ARM_JOINT_ORDER]

    def build_payload(self) -> dict[str, object]:
        return {
            "joint_names": ARM_JOINT_ORDER,
            "positions": self.get_positions(),
        }

    def publish_payload(self, payload: object):
        payload_str = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=True)
        self.publisher.Write(String_(data=payload_str))

    def publish_targets(self):
        self.publish_payload(self.build_payload())

    def publish_joint_targets(self, joint_targets: Mapping[str, float]):
        self.update_joint_targets(joint_targets)
        self.publish_targets()

    def publish_positions(self, positions: list[float]):
        self.set_positions(positions)
        self.publish_targets()
