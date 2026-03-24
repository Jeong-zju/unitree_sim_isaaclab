#!/usr/bin/env python3

import argparse
import time

from g1_upperbody_interface import ARM_JOINT_ORDER, G1UpperBodyPublisher


class UpperBodyKeyboardController:
    def __init__(self, step: float, keyboard_module, network_interface: str = None):
        self.step = step
        self.running = True
        self.selected_joint_index = 0
        self.publisher = G1UpperBodyPublisher(network_interface=network_interface)
        self.listener = keyboard_module.Listener(on_press=self._on_press)
        self.listener.start()
        self._print_help()
        self._print_current_joint()

    def _print_help(self):
        print("=" * 60)
        print("G1 upper-body keyboard control")
        print("[: previous joint    ]: next joint")
        print("J: decrease target   K: increase target")
        print("U: zero current joint")
        print("R: zero all arm joints")
        print("P: print all current targets")
        print("Q: exit")
        print("=" * 60)

    def _print_current_joint(self):
        joint_name = ARM_JOINT_ORDER[self.selected_joint_index]
        joint_value = self.publisher.targets[joint_name]
        print(f"[SELECT] {joint_name} = {joint_value:.3f}")

    def _publish_and_report(self, joint_name: str):
        self.publisher.publish_targets()
        print(f"[TARGET] {joint_name} = {self.publisher.targets[joint_name]:.3f}")

    def _on_press(self, key):
        try:
            key_char = key.char.lower() if hasattr(key, "char") and key.char else None
        except AttributeError:
            key_char = None

        if key_char == "q":
            self.running = False
            return False
        if key_char == "[":
            self.selected_joint_index = (self.selected_joint_index - 1) % len(ARM_JOINT_ORDER)
            self._print_current_joint()
            return
        if key_char == "]":
            self.selected_joint_index = (self.selected_joint_index + 1) % len(ARM_JOINT_ORDER)
            self._print_current_joint()
            return

        joint_name = ARM_JOINT_ORDER[self.selected_joint_index]
        if key_char == "j":
            self.publisher.set_joint_target(
                joint_name,
                self.publisher.targets[joint_name] - self.step,
            )
            self._publish_and_report(joint_name)
        elif key_char == "k":
            self.publisher.set_joint_target(
                joint_name,
                self.publisher.targets[joint_name] + self.step,
            )
            self._publish_and_report(joint_name)
        elif key_char == "u":
            self.publisher.set_joint_target(joint_name, 0.0)
            self._publish_and_report(joint_name)
        elif key_char == "r":
            self.publisher.reset_targets()
            self.publisher.publish_targets()
            print("[TARGET] all arm joints reset to 0.000")
        elif key_char == "p":
            for name in ARM_JOINT_ORDER:
                print(f"{name}: {self.publisher.targets[name]:.3f}")

    def stop(self):
        self.running = False
        if hasattr(self, "listener"):
            self.listener.stop()


def main():
    parser = argparse.ArgumentParser(description="G1 upper-body keyboard command sender")
    parser.add_argument("--step", type=float, default=0.05, help="joint increment step")
    parser.add_argument(
        "--publish-hz",
        type=float,
        default=20.0,
        help="keep publishing the latest arm targets at this frequency",
    )
    parser.add_argument("--network_interface", type=str, default=None, help="DDS network interface, e.g. enp3s0")
    args = parser.parse_args()

    try:
        from pynput import keyboard
    except ImportError:
        print("error: pynput library missing")
        print("please install: pip install pynput")
        return

    controller = UpperBodyKeyboardController(
        step=args.step,
        keyboard_module=keyboard,
        network_interface=args.network_interface,
    )
    publish_interval = 1.0 / max(args.publish_hz, 1e-6)

    try:
        while controller.running:
            controller.publisher.publish_targets()
            time.sleep(publish_interval)
    except KeyboardInterrupt:
        pass
    finally:
        controller.stop()
        controller.publisher.reset_targets()
        controller.publisher.publish_targets()
        print("upper-body command sender stopped")


if __name__ == "__main__":
    main()
