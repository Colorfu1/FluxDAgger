#!/usr/bin/env python3
"""Set every Piper arm in the standard four-arm cell to slave mode via the SDK.

Usage:
    python tools/hardware/piper_set_slave.py

Requires the AgileX Piper SDK. CAN device names match ``dagger.launch`` /
``can_config_modified.sh``. If an arm was in master mode, power-cycle it after
running this script if the switch does not take effect.
"""

from __future__ import annotations
import time
from typing import NamedTuple

from piper_sdk import C_PiperInterface_V2


class ArmCAN(NamedTuple):
    """Arm role (FluxDAgger naming) and its CAN interface."""

    role: str
    can_if: str


ARMS_IN_ORDER = (
    ArmCAN('front_left', 'c_left_slave'),
    ArmCAN('front_right', 'c_right_slave'),
    ArmCAN('rear_left', 'c_left_master'),
    ArmCAN('rear_right', 'c_right_master'),
)

# Piper SDK: 0xFC = slave, 0xFA = master.
SLAVE_MODE_ARGS = (0xFC, 0, 0, 0)

if __name__ == '__main__':
    settle_s = 0.1

    for arm in ARMS_IN_ORDER:
        piper = C_PiperInterface_V2(arm.can_if)

        piper.ConnectPort()
        for _ in range(2):
            piper.MasterSlaveConfig(*SLAVE_MODE_ARGS)
            time.sleep(settle_s)
        piper.ConnectPort()
        time.sleep(settle_s)

        label = f'{arm.role} ({arm.can_if})'
        while not piper.EnablePiper():
            print(f'Still enabling {label} — keep that arm stationary.')
            time.sleep(0.01)
        print(f'Done: {label} is in slave mode and enabled.')
