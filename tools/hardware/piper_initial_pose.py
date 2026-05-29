#!/usr/bin/env python3
"""Move all four Piper arms to the initial parallel pose.

Usage:
    python tools/hardware/piper_initial_pose.py

Requires the AgileX Piper SDK and CAN interfaces
``c_left_slave``, ``c_right_slave``, ``c_left_master``, ``c_right_master``
(see ``can_config_modified.sh`` / ``dagger.launch``).
"""

from __future__ import annotations
import time
from typing import NamedTuple

from piper_sdk import C_PiperInterface_V2


class ArmCAN(NamedTuple):
    """Arm role (FluxDAgger naming) and its CAN interface."""

    role: str
    can_if: str


# Same order / labels as ``piper_set_slave.py``.
ARMS_IN_ORDER = (
    ArmCAN('front_left', 'c_left_slave'),
    ArmCAN('front_right', 'c_right_slave'),
    ArmCAN('rear_left', 'c_left_master'),
    ArmCAN('rear_right', 'c_right_master'),
)

# Millimetre-degree home [X, Y, Z, RX, RY, RZ, gripper].
# Matches ``arms.home_end_pose`` in default.yaml.
HOME_END_POSE = [57.0, 0.0, 215.0, 0.0, 85.0, 0.0, 0.0]


def set_initial_pose(piper: C_PiperInterface_V2,
                     pose: list[float] | None = None) -> None:
    if pose is None:
        pose = list(HOME_END_POSE)

    piper.GripperCtrl(0, 1000, 0x01, 0)
    factor = 1000
    x = round(pose[0] * factor)
    y = round(pose[1] * factor)
    z = round(pose[2] * factor)
    rx = round(pose[3] * factor)
    ry = round(pose[4] * factor)
    rz = round(pose[5] * factor)
    joint_6 = round(pose[6] * factor)

    piper.MotionCtrl_2(0x01, 0x00, 100, 0x00)
    piper.EndPoseCtrl(x, y, z, rx, ry, rz)
    piper.GripperCtrl(abs(joint_6), 1000, 0x01, 0)
    time.sleep(0.01)


if __name__ == '__main__':
    connected: list[tuple[str, str, C_PiperInterface_V2]] = []

    for arm in ARMS_IN_ORDER:
        iface = C_PiperInterface_V2(arm.can_if)
        iface.ConnectPort()
        connected.append((arm.role, arm.can_if, iface))

    for role, can_if, iface in connected:
        while not iface.EnablePiper():
            time.sleep(0.01)
            print(f'Enabling {role} ({can_if})')

    for role, can_if, iface in connected:
        set_initial_pose(iface)
        print(f'Sent home pose to {role} ({can_if})')
