import ast
import json
from typing import Any, Mapping, Sequence


def _parse_raw_message(raw_message: Any) -> Any:
    if raw_message is None:
        return None
    if isinstance(raw_message, (dict, list, tuple)):
        return raw_message
    if not isinstance(raw_message, str):
        return None

    message = raw_message.strip()
    if not message:
        return None

    try:
        return json.loads(message)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(message)
        except (SyntaxError, ValueError):
            return None


def _zip_joint_positions(
    joint_names: Sequence[str], positions: Sequence[Any]
) -> dict[str, float] | None:
    if len(positions) < len(joint_names):
        return None
    return {joint_names[i]: float(positions[i]) for i in range(len(joint_names))}


def parse_arm_command_message(
    raw_message: Any,
    joint_names: Sequence[str],
) -> dict[str, float] | None:
    parsed = _parse_raw_message(raw_message)
    if parsed is None:
        return None

    if isinstance(parsed, (list, tuple)):
        return _zip_joint_positions(joint_names, parsed)

    if not isinstance(parsed, Mapping):
        return None

    for key in ("positions", "joint_positions"):
        if key not in parsed:
            continue

        positions = parsed.get(key)
        if not isinstance(positions, (list, tuple)):
            return None

        names = parsed.get("joint_names", joint_names)
        if isinstance(names, (list, tuple)):
            joint_map = {}
            for name, value in zip(names, positions):
                if name in joint_names:
                    joint_map[str(name)] = float(value)
            return joint_map or None

        return _zip_joint_positions(joint_names, positions)

    joint_map = {}
    for joint_name in joint_names:
        if joint_name in parsed:
            joint_map[joint_name] = float(parsed[joint_name])
    return joint_map or None


def apply_arm_command_message(
    raw_message: Any,
    joint_names: Sequence[str],
    arm_joint_name_to_local_index: Mapping[str, int],
    arm_command_state,
) -> bool:
    joint_targets = parse_arm_command_message(raw_message, joint_names)
    if not joint_targets:
        return False

    for joint_name, target in joint_targets.items():
        local_index = arm_joint_name_to_local_index.get(joint_name)
        if local_index is None:
            continue
        arm_command_state[local_index] = float(target)
    return True
