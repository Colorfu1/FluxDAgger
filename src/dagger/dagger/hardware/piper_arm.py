#!/usr/bin/env python3
# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Single-arm Piper control wrapper.

The class keeps SDK access local to one arm. Topics are provided by outer ROS
nodes so data collection and inference can switch command sources without
reconfiguring the hardware layer.

Example:
    arm = PiperArm(
        can_port="c_left_master",
        init_mode="slave",
        joint_pub_topic="/master/joint/left_human",
        joint_sub_topic="/master/joint/left",
    )
    arm.publish_as_master()
"""

from __future__ import annotations
import math
import time
from typing import Optional

import rospy
from geometry_msgs.msg import PoseStamped
from piper_sdk import C_PiperInterface_V2
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from tf.transformations import quaternion_from_euler

from dagger.infra.dagger_log import get_logger


class PiperArm:
    """Control one Piper arm with configurable ROS topics."""

    def __init__(
        self,
        can_port: str,
        *,
        init_mode: str = 'slave',  # "master" / "slave"
        joint_pub_topic: Optional[str] = None,
        joint_sub_topic: str = '/master/joint_arm',
        # Optional observed joint-state topic, published in any mode.
        joint_state_pub_topic: Optional[str] = None,
        # Optional observed end-pose topic, published in any mode.
        end_pose_pub_topic: Optional[str] = None,
        gripper_exist: bool = True,
        name: str = '',
    ) -> None:
        """
        Args:
            can_port: CAN interface name, for example ``c_left_master``.
            init_mode: Initial mode, ``master`` or ``slave``.
            joint_pub_topic: JointState topic published in master mode.
            joint_sub_topic: JointState topic subscribed in slave mode.
            joint_state_pub_topic: Optional observed joint-state topic.
            end_pose_pub_topic: Optional observed end-pose topic.
            gripper_exist: Whether the arm has a gripper.
            name: Arm name used in logs.
        """
        self.can_port = can_port
        self.mode = init_mode
        self.gripper_exist = gripper_exist
        self.name = name if name else can_port
        self._log = get_logger(__name__, prefix=self.name)

        # Topic configuration.
        self.joint_pub_topic = joint_pub_topic
        self.joint_sub_topic = joint_sub_topic
        self.joint_state_pub_topic = joint_state_pub_topic
        self.end_pose_pub_topic = end_pose_pub_topic

        # SDK connection.
        self.piper = C_PiperInterface_V2(can_name=self.can_port)
        self.piper.ConnectPort()

        # Master-mode command publisher. Front arms do not need one.
        self.joint_pub = None
        if self.joint_pub_topic:
            self.joint_pub = rospy.Publisher(
                self.joint_pub_topic, JointState, queue_size=10)

        # Observed joint-state publisher.
        self.joint_state_pub = None
        if self.joint_state_pub_topic:
            self.joint_state_pub = rospy.Publisher(
                self.joint_state_pub_topic, JointState, queue_size=10)

        # Observed end-pose publisher.
        self.end_pose_pub = None
        if self.end_pose_pub_topic:
            # Subscribers expect PoseStamped.
            self.end_pose_pub = rospy.Publisher(
                self.end_pose_pub_topic, PoseStamped, queue_size=10)

        # Slave command subscriber.
        self.sub_joint = None

        # Interpolate slave commands when jumps exceed this degree threshold.
        self.qpos_interpolation_threshold_deg = 5.0

        self.last_slave_command = None
        self.slave_interpolation_target = None

        # Initialize SDK master/slave mode.
        if init_mode == 'master':
            self.set_master()
        else:
            self.set_slave()
            # Subscription is controlled explicitly by ArmNode.
            # if self.joint_sub_topic:
            #     self.set_joint_sub_topic(self.joint_sub_topic)

    def unsubscribe_joint(self):
        """Unsubscribe from the current slave JointState topic."""
        if self.sub_joint:
            try:
                self.sub_joint.unregister()
                self._log.info('Unsubscribed joint topic')
            except AssertionError as e:
                # Another thread may have already unregistered this subscriber.
                self._log.warning('Concurrent joint unsubscribe failed: %s', e)
            finally:
                self.sub_joint = None

    def set_joint_sub_topic(self, topic: str):
        """Set and subscribe to the slave JointState topic.

        Args:
            topic: JointState topic name.
        """
        self.joint_sub_topic = topic
        if self.mode == 'slave':
            if self.sub_joint:
                self.sub_joint.unregister()
            # Reset interpolation state when switching command sources.
            self.last_slave_command = None
            self.slave_interpolation_target = None
            self.sub_joint = rospy.Subscriber(
                self.joint_sub_topic,
                JointState,
                self.joint_slave_callback,
                queue_size=10,
            )
            self._log.info('Subscribed joint topic: %s', self.joint_sub_topic)
        else:
            self._log.warning(
                'Cannot subscribe joint topic in master mode; switch to slave '
                'mode first')

    def set_master(self):
        """Switch SDK mode to master and drop any slave subscription.

        Subscriptions are not restored automatically when switching back to
        slave; callers must invoke :meth:`set_joint_sub_topic`.
        """
        self._log.warning('Switching to master mode')
        self.mode = 'master'

        # Repeat SDK mode setup to improve reliability.
        for i in range(2):
            self.piper.MasterSlaveConfig(0xFA, 0, 0, 0)

            time.sleep(0.1)

        self.unsubscribe_joint()

    def set_slave(self):
        """Switch SDK mode to slave without subscribing automatically.

        Callers decide which command topic to subscribe next.
        """
        self._log.warning('Switching to slave mode')
        self.mode = 'slave'

        # Repeat SDK mode setup to improve reliability.
        for i in range(2):
            self.piper.MasterSlaveConfig(0xFC, 0, 0, 0)

            time.sleep(0.1)

    def publish_as_master(self):
        """Publish current SDK joint/gripper command state in master mode.

        Smoothing is handled on the slave side, so this method forwards the SDK
        state directly.
        """
        if self.mode != 'master' or self.joint_pub is None:
            return

        js = JointState()
        js.header.stamp = rospy.Time.now()
        js.name = [
            'joint0',
            'joint1',
            'joint2',
            'joint3',
            'joint4',
            'joint5',
            'joint6',
        ]

        # Match rear_arm_control.py by reading the current control target.
        joint_ctrl = self.piper.GetArmJointCtrl().joint_ctrl
        gripper_ctrl = self.piper.GetArmGripperCtrl().gripper_ctrl

        # SDK joint values are millidegrees; convert to radians.
        current_pos = [
            (joint_ctrl.joint_1 / 1000.0) * 0.017444,
            (joint_ctrl.joint_2 / 1000.0) * 0.017444,
            (joint_ctrl.joint_3 / 1000.0) * 0.017444,
            (joint_ctrl.joint_4 / 1000.0) * 0.017444,
            (joint_ctrl.joint_5 / 1000.0) * 0.017444,
            (joint_ctrl.joint_6 / 1000.0) * 0.017444,
            gripper_ctrl.grippers_angle / 1000000.0,
        ]

        js.position = current_pos
        self.joint_pub.publish(js)

    def publish_joint_state(self):
        """Publish observed joint state when a state topic is configured.

        This path is used by collection and inference consumers.
        """
        if not self.joint_state_pub:
            return

        js = JointState()
        js.header.stamp = rospy.Time.now()
        js.name = [
            'joint0',
            'joint1',
            'joint2',
            'joint3',
            'joint4',
            'joint5',
            'joint6',
        ]

        # Read observed state from the SDK.
        joint_msg = self.piper.GetArmJointMsgs().joint_state
        gripper_msg = self.piper.GetArmGripperMsgs().gripper_state
        # SDK joint values are millidegrees; convert to radians.
        js.position = [
            (joint_msg.joint_1 / 1000.0) * 0.017444,
            (joint_msg.joint_2 / 1000.0) * 0.017444,
            (joint_msg.joint_3 / 1000.0) * 0.017444,
            (joint_msg.joint_4 / 1000.0) * 0.017444,
            (joint_msg.joint_5 / 1000.0) * 0.017444,
            (joint_msg.joint_6 / 1000.0) * 0.017444,
            gripper_msg.grippers_angle / 1000000.0,
        ]
        # Match piper_ctrl_single_node.py velocity/effort layout.
        high_spd_msg = self.piper.GetArmHighSpdInfoMsgs()
        vel_0 = high_spd_msg.motor_1.motor_speed / 1000.0
        vel_1 = high_spd_msg.motor_2.motor_speed / 1000.0
        vel_2 = high_spd_msg.motor_3.motor_speed / 1000.0
        vel_3 = high_spd_msg.motor_4.motor_speed / 1000.0
        vel_4 = high_spd_msg.motor_5.motor_speed / 1000.0
        vel_5 = high_spd_msg.motor_6.motor_speed / 1000.0
        js.velocity = [vel_0, vel_1, vel_2, vel_3, vel_4, vel_5, 0.0]

        effort_6 = gripper_msg.grippers_effort / 1000.0
        js.effort = [0, 0, 0, 0, 0, 0, effort_6]

        self.joint_state_pub.publish(js)

    def publish_end_pose(self):
        """Publish observed end pose when an end-pose topic is configured.

        This path is used by collection and inference consumers.
        """
        if not self.end_pose_pub:
            return

        end_pose_msg = self.piper.GetArmEndPoseMsgs().end_pose

        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = rospy.Time.now()
        pose_stamped.header.frame_id = 'base_link'

        # Position is reported in micrometers.
        pose_stamped.pose.position.x = end_pose_msg.X_axis / 1000000
        pose_stamped.pose.position.y = end_pose_msg.Y_axis / 1000000
        pose_stamped.pose.position.z = end_pose_msg.Z_axis / 1000000

        # Keep the same Euler convention as piper_ctrl_single_node.py.
        roll = end_pose_msg.RX_axis / 1000
        pitch = end_pose_msg.RY_axis / 1000
        yaw = end_pose_msg.RZ_axis / 1000
        quaternion = quaternion_from_euler(roll, pitch, yaw)

        pose_stamped.pose.orientation.x = quaternion[0]
        pose_stamped.pose.orientation.y = quaternion[1]
        pose_stamped.pose.orientation.z = quaternion[2]
        pose_stamped.pose.orientation.w = quaternion[3]

        self.end_pose_pub.publish(pose_stamped)

    def joint_slave_callback(self, joint_data: JointState):
        """Forward slave JointState commands to the SDK.

        Large jumps are interpolated against the previous command to reduce
        discontinuities in executed motion and recorded feedback.
        """
        if self.mode != 'slave':
            return

        target_pos = list(joint_data.position)

        if self.last_slave_command is None:
            self.last_slave_command = target_pos.copy()
            actual_pos = target_pos
        else:
            # Use the previous command as the interpolation reference.
            reference_pos = self.last_slave_command

            max_diff_deg = 0.0
            for i in range(6):
                diff_rad = abs(target_pos[i] - reference_pos[i])
                diff_deg = diff_rad * 180.0 / math.pi
                max_diff_deg = max(max_diff_deg, diff_deg)

            if max_diff_deg > self.qpos_interpolation_threshold_deg:
                if self.slave_interpolation_target is None:
                    self.slave_interpolation_target = target_pos

                target_changed = False
                for i in range(7):
                    if abs(target_pos[i] -
                           self.slave_interpolation_target[i]) > 1e-6:
                        target_changed = True
                        break

                if target_changed:
                    self.slave_interpolation_target = target_pos

                # Move one bounded interpolation step toward the target.
                step_size_rad = (
                    self.qpos_interpolation_threshold_deg * math.pi / 180.0)
                interpolated_pos = []
                for i in range(7):
                    diff = (
                        self.slave_interpolation_target[i] - reference_pos[i])
                    if (i < 6 and abs(diff) * 180.0 / math.pi >
                            self.qpos_interpolation_threshold_deg):
                        step = step_size_rad if diff > 0 else -step_size_rad
                        interpolated_pos.append(reference_pos[i] + step)
                    else:
                        interpolated_pos.append(
                            self.slave_interpolation_target[i])

                actual_pos = interpolated_pos
            else:
                actual_pos = target_pos
                self.slave_interpolation_target = None

        self.last_slave_command = (
            actual_pos.copy()
            if isinstance(actual_pos, list) else list(actual_pos))

        # Convert to SDK units and issue the command.
        factor = 180.0 / math.pi * 1000.0
        joint_angles = [round(actual_pos[i] * factor) for i in range(6)]
        gripper_angle = round(actual_pos[6] * 1e6)
        gripper_angle = max(0, min(80000, gripper_angle))

        self.piper.MotionCtrl_2(0x01, 0x01, 100)
        self.piper.JointCtrl(*joint_angles)
        if self.gripper_exist:
            self.piper.GripperCtrl(gripper_angle, 1000, 0x01, 0)

    def enable_callback(self, enable_flag: Bool):
        """Enable or disable the arm regardless of master/slave mode."""
        self._log.info('enable_flag: %s', enable_flag.data)
        if enable_flag.data:
            self.piper.EnableArm(7)
            if self.gripper_exist:
                self.piper.GripperCtrl(0, 1000, 0x01, 0)
        else:
            self.piper.DisableArm(7)
            if self.gripper_exist:
                self.piper.GripperCtrl(0, 1000, 0x00, 0)

    def piper_ctl_joint(self, joint_angles):
        """Move the arm to a target joint configuration.

        Args:
            joint_angles: Seven values ``[joint0, ..., joint6]`` in radians.
        """
        if joint_angles is None or len(joint_angles) < 7:
            self._log.warning(
                'piper_ctl_joint: invalid joint_angles, expected len>=7')
            return

        factor = 180.0 / math.pi * 1000.0
        joint = [round(joint_angles[i] * factor) for i in range(6)]
        gripper_angle = round(joint_angles[6] * 1e6)

        self.piper.MotionCtrl_2(0x01, 0x01, 100, 0x00)
        joint0, joint1, joint2, joint3, joint4, joint5 = joint
        self.piper.JointCtrl(joint0, joint1, joint2, joint3, joint4, joint5)
        if self.gripper_exist:
            self.piper.GripperCtrl(abs(gripper_angle), 1000, 0x01, 0)

    def piper_ctl_end_pose(self, pose):
        """Move the arm to a target end pose through the SDK.

        Args:
            pose: ``[X, Y, Z, RX, RY, RZ, gripper]`` in the demo SDK units.
        """
        if pose is None or len(pose) < 7:
            self._log.warning(
                'piper_ctl_end_pose: invalid pose, expected len>=7')
            return

        factor = 1000
        position = pose

        X = round(position[0] * factor)
        Y = round(position[1] * factor)
        Z = round(position[2] * factor)
        RX = round(position[3] * factor)
        RY = round(position[4] * factor)
        RZ = round(position[5] * factor)
        joint_6 = round(position[6] * factor)

        # Enable the gripper channel before issuing the end-pose command.
        if self.gripper_exist:
            self.piper.GripperCtrl(0, 1000, 0x01, 0)
        # Keep the V1-compatible SDK call sequence.
        self.piper.MotionCtrl_2(0x01, 0x00, 100)
        self.piper.EndPoseCtrl(X, Y, Z, RX, RY, RZ)
        if self.gripper_exist:
            self.piper.GripperCtrl(abs(joint_6), 1000, 0x01, 0)
