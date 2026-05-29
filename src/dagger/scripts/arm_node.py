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
"""arm_node.py.

Single-arm ROS node that handles control commands, arm status publishing, and
master/slave mode switching.
"""

from __future__ import annotations
import sys
import threading
import traceback

import rospy
from std_msgs.msg import Bool, String

from dagger.hardware.piper_arm import PiperArm
from dagger.infra.dagger_log import get_logger
from dagger.runtime_config import (get_ros_override, get_ros_param,
                                   load_runtime_config_from_ros)
from dagger_msgs.msg import ArmStatus, GlobalState

logger = get_logger(__name__, prefix='ArmNode')


class ArmNode:
    """Single-arm ROS node."""

    def __init__(self):
        """Initialize the arm node."""
        self.runtime_config = load_runtime_config_from_ros('~')
        runtime = self.runtime_config.runtime
        self.arm_name = get_ros_param('arm_name', 'unknown')

        try:
            self.profile = self.runtime_config.arm_profiles[self.arm_name]
        except KeyError:
            logger.error(f'Unknown arm_name: {self.arm_name}')
            sys.exit(1)

        self.can_port = get_ros_param('can_port', self.profile.can_port)
        self.puppet_publish_rate = get_ros_override(
            'puppet_publish_rate',
            runtime.puppet_publish_rate,
            value_type=float,
        )
        self.master_publish_rate = get_ros_override(
            'master_publish_rate',
            runtime.master_publish_rate,
            value_type=float,
        )

        self.is_front_arm = self.profile.is_front
        self.is_rear_arm = self.profile.is_rear
        topics = self.runtime_config.topics

        self.current_mode = 'slave'
        self.subscribed_topic = ''
        self.is_enabled = True
        self.is_human_mode = False
        self.is_ready = False

        logger.info(f'[ArmNode][{self.arm_name}] '
                    f'Initializing arm on CAN port: {self.can_port}')

        self.arm = PiperArm(
            can_port=self.can_port,
            init_mode='slave',
            joint_pub_topic=self.profile.joint_pub_topic,
            joint_sub_topic=self.profile.joint_sub_topic,
            joint_state_pub_topic=self.profile.joint_state_pub_topic,
            end_pose_pub_topic=self.profile.end_pose_pub_topic,
            name=self.arm_name,
        )

        try:
            _ = self.arm.piper.GetArmJointMsgs()
            self.is_ready = True
            logger.info(f'[ArmNode][{self.arm_name}] SDK connection OK')
        except Exception as e:
            logger.warning(
                f'[ArmNode][{self.arm_name}] SDK connection check failed: {e}')
            self.is_ready = False

        self.status_pub = rospy.Publisher(
            topics.arm_status(self.arm_name),
            ArmStatus,
            queue_size=10,
            latch=True)

        self.control_cmd_sub = rospy.Subscriber(
            topics.dagger_control_command,
            String,
            self.control_cmd_callback,
            queue_size=10)

        self.mode_cmd_sub = rospy.Subscriber(
            topics.arm_mode(self.arm_name),
            String,
            self.mode_cmd_callback,
            queue_size=10)

        self.subscribe_cmd_sub = rospy.Subscriber(
            topics.arm_subscribe(self.arm_name),
            String,
            self.subscribe_cmd_callback,
            queue_size=10)

        self.enable_cmd_sub = rospy.Subscriber(
            topics.arm_enable(self.arm_name),
            Bool,
            self.enable_cmd_callback,
            queue_size=10)

        self.global_state_sub = rospy.Subscriber(
            topics.dagger_global_state,
            GlobalState,
            self.global_state_callback,
            queue_size=10)

        self.status_thread = threading.Thread(
            target=self.publish_status_loop,
            daemon=True,
            name=f'{self.arm_name}_status')
        self.status_thread.start()

        if self.is_front_arm:
            self.puppet_pub_thread = threading.Thread(
                target=self.publish_joint_state_loop,
                daemon=True,
                name=f'{self.arm_name}_puppet_pub')
            self.puppet_pub_thread.start()

        if self.is_rear_arm:
            self.master_pub_thread = threading.Thread(
                target=self.publish_master_cmd_loop,
                daemon=True,
                name=f'{self.arm_name}_master_pub')
            self.master_pub_thread.start()

        logger.info(f'[ArmNode][{self.arm_name}] Node started successfully')

    def control_cmd_callback(self, msg):
        """Handle global control commands."""
        cmd = msg.data
        logger.info(
            f'[ArmNode][{self.arm_name}] Received control command: {cmd}')

        if cmd == 'start_collect':
            self.handle_start_collect()
        elif cmd == 'stop_collect':
            self.handle_stop_collect()
        elif cmd == 'human_mode':
            self.handle_human_mode()
        elif cmd == 'inference_mode':
            self.handle_inference_mode()
        elif cmd == 'move_home':
            self.handle_move_home(use_end_pose=False)
        elif cmd == 'move_home_endpose':
            self.handle_move_home(use_end_pose=True)
        elif cmd == 'quit':
            logger.info(f'[ArmNode][{self.arm_name}] '
                        f'Received quit command, shutting down node')
            rospy.signal_shutdown('Quit command received from controller')
        else:
            logger.warning(
                f'[ArmNode][{self.arm_name}] Unknown control command: {cmd}')

    def mode_cmd_callback(self, msg):
        """Handle arm mode switch commands.

        Args:
            msg: String message containing ``master`` or ``slave``.
        """
        mode = msg.data
        logger.info(
            f'[ArmNode][{self.arm_name}] Received mode command: {mode}')

        if mode == 'master':
            self.arm.set_master()
            self.current_mode = 'master'
            logger.info(f'[ArmNode][{self.arm_name}] Switched to MASTER mode')
        elif mode == 'slave':
            self.arm.set_slave()
            self.current_mode = 'slave'
            logger.info(f'[ArmNode][{self.arm_name}] Switched to SLAVE mode')
        else:
            logger.warning(f'[ArmNode][{self.arm_name}] Unknown mode: {mode}')

    def subscribe_cmd_callback(self, msg):
        """Handle target subscription switch commands.

        Args:
            msg: String message with the target topic, or empty to unsubscribe.
        """
        topic = msg.data
        logger.info(
            f'[ArmNode][{self.arm_name}] Received subscribe command: {topic}')

        if topic:
            self.arm.set_joint_sub_topic(topic)
            self.subscribed_topic = topic
        else:
            self.arm.unsubscribe_joint()
            self.subscribed_topic = ''

    def enable_cmd_callback(self, msg):
        """Handle arm enable commands.

        Args:
            msg: Bool message where ``True`` enables the arm.
        """
        self.is_enabled = msg.data
        logger.info(f'[ArmNode][{self.arm_name}] Enable: {self.is_enabled}')
        self.arm.enable_callback(msg)

    def global_state_callback(self, msg):
        """Handle global state updates."""
        self.is_human_mode = msg.is_human_mode

    def handle_start_collect(self):
        """Handle start-collection commands."""
        logger.info(f'[ArmNode][{self.arm_name}] Handling start_collect')

    def handle_stop_collect(self):
        """Handle stop-collection commands."""
        logger.info(f'[ArmNode][{self.arm_name}] Handling stop_collect')
        self.subscribed_topic = ''

    def handle_human_mode(self):
        """Handle human teleoperation mode."""
        logger.info(f'[ArmNode][{self.arm_name}] Handling human_mode')

    def handle_inference_mode(self):
        """Handle inference mode."""
        logger.info(f'[ArmNode][{self.arm_name}] Handling inference_mode')

    def handle_move_home(self, use_end_pose=False):
        """Handle home-position commands."""
        logger.info(f'[ArmNode][{self.arm_name}] '
                    f'Moving to home (use_end_pose={use_end_pose})')

        home_position_end_pose = self.profile.home_end_pose
        home_position_joint = self.profile.home_joint
        if not home_position_joint:
            logger.error(f'[ArmNode][{self.arm_name}] '
                         f'Home joint position is not configured')
            return

        try:
            if use_end_pose:
                logger.info(f'[ArmNode][{self.arm_name}] '
                            f'Moving to home using end pose control')
                self.arm.piper_ctl_end_pose(home_position_end_pose)
            else:
                logger.info(f'[ArmNode][{self.arm_name}] '
                            f'Moving to home using joint control')
                self.arm.piper_ctl_joint(home_position_joint)

            logger.info(
                f'[ArmNode][{self.arm_name}] Home position command sent')
        except Exception as e:
            logger.error('[ArmNode][%s] Error moving to home: %s\n%s',
                         self.arm_name, e, traceback.format_exc())

    def publish_status_loop(self):
        """Publish arm status at 10 Hz.

        The status publisher is latched so new subscribers immediately receive
        the latest status.
        """
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            status = ArmStatus()
            status.arm_name = self.arm_name
            status.current_mode = self.current_mode
            status.subscribed_topic = self.subscribed_topic
            status.is_enabled = self.is_enabled
            status.is_ready = self.is_ready
            self.status_pub.publish(status)
            rate.sleep()

    def publish_joint_state_loop(self):
        """Publish front-arm joint state and end-effector pose."""
        if not self.is_front_arm:
            return

        rate = rospy.Rate(self.puppet_publish_rate)
        while not rospy.is_shutdown():
            try:
                self.arm.publish_joint_state()
                self.arm.publish_end_pose()
            except Exception as e:
                logger.warning(f'[ArmNode][{self.arm_name}] '
                               f'Error in publish_joint_state: {e}')
            rate.sleep()

    def publish_master_cmd_loop(self):
        """Publish rear-arm master commands while in human mode."""
        if not self.is_rear_arm:
            return

        rate = rospy.Rate(self.master_publish_rate)
        while not rospy.is_shutdown():
            try:
                if self.is_human_mode and self.current_mode == 'master':
                    self.arm.publish_as_master()
            except Exception as e:
                logger.warning(f'[ArmNode][{self.arm_name}] '
                               f'Error in publish_master_cmd: {e}')
            rate.sleep()

    def shutdown(self):
        """Clean up resources on node shutdown."""
        logger.info(f'[ArmNode][{self.arm_name}] '
                    f'Shutting down and cleaning up resources...')


def main():
    """Initialize ROS and start ``ArmNode``."""
    rospy.init_node('arm_node', anonymous=True)

    arm_name = get_ros_param('arm_name', 'unknown')

    if arm_name != 'unknown':
        logger.info(f'Starting node for arm: {arm_name}')

    try:
        node = ArmNode()

        rospy.on_shutdown(node.shutdown)

        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        logger.error('Fatal error: %s\n%s', e, traceback.format_exc())


if __name__ == '__main__':
    main()
