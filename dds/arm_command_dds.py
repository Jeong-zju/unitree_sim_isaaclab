# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""
Simple upper-body command DDS interface.
"""

import json
from typing import Any, Dict, Optional

from dds.dds_base import DDSObject
from unitree_sdk2py.core.channel import ChannelSubscriber
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_


class ArmCommandDDS(DDSObject):
    """Receive arm joint targets from a simple string topic."""

    def __init__(self, node_name: str = "arm_command"):
        if hasattr(self, "_initialized"):
            return

        super().__init__()
        self._initialized = True
        self.node_name = node_name
        self.setup_shared_memory(
            output_shm_name="isaac_arm_command_cmd",
            output_size=2048,
            inputshm_flag=False,
            outputshm_flag=True,
        )
        self.write_arm_command({})
        print(f"[{self.node_name}] Arm command DDS node initialized")

    def setup_publisher(self) -> bool:
        return True

    def setup_subscriber(self) -> bool:
        try:
            self.subscriber = ChannelSubscriber("rt/arm_command/cmd", String_)
            self.subscriber.Init(lambda msg: self.dds_subscriber(msg, ""), 1)
            print(f"[{self.node_name}] Arm command subscriber initialized")
            return True
        except Exception as e:
            print(f"arm_command_dds [{self.node_name}] Failed to initialize subscriber: {e}")
            return False

    def dds_publisher(self) -> Any:
        return None

    def dds_subscriber(self, msg: String_, datatype: str = None) -> Dict[str, Any]:
        try:
            cmd_data = {"arm_command": msg.data}
            self.output_shm.write_data(cmd_data)
            return cmd_data
        except Exception as e:
            print(f"arm_command_dds [{self.node_name}] Failed to process subscribe data: {e}")
            return {}

    def get_arm_command(self) -> Optional[Dict[str, Any]]:
        if self.output_shm:
            return self.output_shm.read_data()
        return None

    def write_arm_command(self, command: Any):
        try:
            if not isinstance(command, str):
                command = json.dumps(command, ensure_ascii=True)
            if self.output_shm:
                self.output_shm.write_data({"arm_command": command})
        except Exception as e:
            print(f"arm_command_dds [{self.node_name}] Failed to write arm command: {e}")
