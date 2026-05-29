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
"""dagger_controller_node.py.

Control node for keyboard input, global DAgger state, and arm coordination.
"""

from __future__ import annotations
import select
import sys
import termios
import threading
import time
import traceback
import tty
from functools import partial

import rospy
from std_msgs.msg import Bool, String

from dagger.infra.dagger_log import get_logger
from dagger.runtime_config import load_runtime_config_from_ros
from dagger_msgs.msg import ArmStatus, GlobalState, InputRequest, InputResult

logger = get_logger(__name__, prefix='DaggerController')


class DaggerController:
    """DAgger control node."""

    def __init__(self):
        """Initialize publishers, subscribers, and the keyboard thread."""
        self.runtime_config = load_runtime_config_from_ros('~')
        self.arm_profiles = self.runtime_config.arm_profiles
        self.topics = self.runtime_config.topics
        self.front_arm_names = [
            name for name, profile in self.arm_profiles.items()
            if profile.is_front
        ]
        self.rear_arm_names = [
            name for name, profile in self.arm_profiles.items()
            if profile.is_rear
        ]
        topics = self.topics

        self.is_collecting = False
        self.is_human_mode = False
        self.is_episode_active = False
        self.is_running = True

        self.arm_statuses = {}
        self.arm_status_lock = threading.Lock()

        self.control_cmd_pub = rospy.Publisher(
            topics.dagger_control_command, String, queue_size=10, latch=True)

        self.global_state_pub = rospy.Publisher(
            topics.dagger_global_state, GlobalState, queue_size=10, latch=True)

        self.collector_cmd_pub = rospy.Publisher(
            topics.dagger_collector_command, String, queue_size=10, latch=True)

        self.input_request_sub = rospy.Subscriber(
            topics.dagger_input_request,
            InputRequest,
            self.handle_input_request,
            queue_size=10)

        self.input_result_pub = rospy.Publisher(
            topics.dagger_input_result, InputResult, queue_size=10)

        self.pending_input_request = None
        self.input_request_lock = threading.Lock()

        self.arm_names = list(self.arm_profiles.keys())
        self.arm_mode_pubs = {}
        self.arm_subscribe_pubs = {}
        self.arm_enable_pubs = {}

        for arm_name in self.arm_names:
            self.arm_mode_pubs[arm_name] = rospy.Publisher(
                topics.arm_mode(arm_name), String, queue_size=10, latch=True)
            self.arm_subscribe_pubs[arm_name] = rospy.Publisher(
                topics.arm_subscribe(arm_name),
                String,
                queue_size=10,
                latch=True)
            self.arm_enable_pubs[arm_name] = rospy.Publisher(
                topics.arm_enable(arm_name), Bool, queue_size=10, latch=True)

        self.arm_status_subs = {}
        for arm_name in self.arm_names:
            self.arm_status_subs[arm_name] = rospy.Subscriber(
                topics.arm_status(arm_name),
                ArmStatus,
                partial(self.arm_status_callback, arm_name),
                queue_size=10)

        logger.info('Waiting for arm nodes to be ready...')
        max_wait_time = 5.0
        wait_start = time.time()
        all_ready = False
        while (time.time() - wait_start) < max_wait_time:
            with self.arm_status_lock:
                ready_count = sum(1 for status in self.arm_statuses.values()
                                  if status.get('is_ready', False))
                if ready_count == len(self.arm_names):
                    all_ready = True
                    break
            rospy.sleep(0.1)

        if all_ready:
            logger.info(f'All {len(self.arm_names)} '
                        f'arm nodes are ready')
        else:
            logger.warning(f'Some arm nodes may not be ready '
                           f'after {max_wait_time}s wait')

        self.publish_global_state()

        logger.info('Enabling all arms...')
        for arm_name in self.arm_names:
            self.arm_enable_pubs[arm_name].publish(Bool(True))
            rospy.sleep(0.1)

        logger.info('Sending repeated '
                    'move_home commands to all arms...')
        for i in range(3):
            logger.info(f'move_home attempt {i+1}/3')
            self.control_cmd_pub.publish(String('move_home'))
            rospy.sleep(2.0)

        self.keyboard_thread = threading.Thread(
            target=self.keyboard_loop, daemon=True, name='keyboard_loop')
        self.keyboard_thread.start()

        logger.info('Node started. Keys: '
                    's=start, h=human, i=inference, r=stop, d=discard, q=quit')

    def arm_status_callback(self, arm_name, msg):
        """Handle arm status updates.

        Args:
            arm_name: Arm name.
            msg: ``ArmStatus`` message.
        """
        with self.arm_status_lock:
            self.arm_statuses[arm_name] = {
                'current_mode': msg.current_mode,
                'subscribed_topic': msg.subscribed_topic,
                'is_enabled': msg.is_enabled,
                'is_ready': msg.is_ready,
            }

    def publish_global_state(self):
        """Publish the current global DAgger state."""
        state = GlobalState()
        state.is_collecting = self.is_collecting
        state.is_human_mode = self.is_human_mode
        state.is_episode_active = self.is_episode_active
        state.current_mode = 'human' if self.is_human_mode else 'inference'
        self.global_state_pub.publish(state)

    def _set_arms_subscribe_inference(self):
        """Route all arms to the model-output topic."""
        for arm_name in self.arm_names:
            profile = self.arm_profiles[arm_name]
            self.arm_subscribe_pubs[arm_name].publish(
                String(profile.joint_sub_topic))

    def _set_front_arms_subscribe_human(self):
        """Route front arms to rear-arm teleoperation topics."""
        for arm_name in self.front_arm_names:
            profile = self.arm_profiles[arm_name]
            if profile.human_sub_topic:
                self.arm_subscribe_pubs[arm_name].publish(
                    String(profile.human_sub_topic))

    def handle_input_request(self, msg):
        """Handle an operator input request.

        Existing pending requests are cancelled before the new request becomes
        active. Actual key reads happen in ``keyboard_loop``.

        Args:
            msg: ``InputRequest`` message.
        """
        with self.input_request_lock:
            if self.pending_input_request is not None:
                old_request_id = self.pending_input_request.request_id
                result = InputResult()
                result.request_id = old_request_id
                result.input = ''
                result.success = False
                result.error_message = (
                    'Request cancelled: new request received')
                self.input_result_pub.publish(result)
                logger.warning(f'Cancelled previous input request: '
                               f'{old_request_id}')

            self.pending_input_request = msg
            logger.info(f'Received input request: {msg.request_id}')
            logger.info('%s', '=' * 60)
            if msg.prompt in (
                    'Episode Success? [1=success / 0=failure] : ',
                    'Episode Score (1-5): ',
            ):
                logger.info_yellow('%s', msg.prompt)
            else:
                logger.info('%s', msg.prompt)
            logger.info('%s', '=' * 60)

    def keyboard_loop(self):
        """Run the keyboard input loop.

        Handles collection hotkeys and pending collector input requests.
        """
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while not rospy.is_shutdown() and self.is_running:
                pending_request = None
                with self.input_request_lock:
                    pending_request = self.pending_input_request

                if select.select([sys.stdin], [], [], 0.1)[0]:
                    ch = sys.stdin.read(1)

                    if pending_request is not None:
                        valid_chars = pending_request.valid_chars

                        if ch in valid_chars or ch.lower(
                        ) in valid_chars.lower():
                            with self.input_request_lock:
                                if (self.pending_input_request is not None and
                                        self.pending_input_request.request_id
                                        == pending_request.request_id):
                                    self.pending_input_request = None

                            result = InputResult()
                            result.request_id = pending_request.request_id
                            result.input = ch
                            result.success = True
                            result.error_message = ''

                            self.input_result_pub.publish(result)
                            logger.info('Input: %s', ch)

                            continue
                        elif ch == '\x03':  # Ctrl+C
                            with self.input_request_lock:
                                if (self.pending_input_request is not None and
                                        self.pending_input_request.request_id
                                        == pending_request.request_id):
                                    self.pending_input_request = None

                            result = InputResult()
                            result.request_id = pending_request.request_id
                            result.input = ''
                            result.success = False
                            result.error_message = 'Input cancelled by user'

                            self.input_result_pub.publish(result)

                            continue
                        logger.warning(
                            'Invalid input: "%s". '
                            'Expected one of: [%s]',
                            ch,
                            valid_chars,
                        )
                        logger.info_yellow('%s', pending_request.prompt)
                        continue

                    if ch in ('s', 'S'):
                        self.handle_start_collect()
                    elif ch in ('h', 'H'):
                        self.handle_human_mode()
                    elif ch in ('i', 'I'):
                        self.handle_inference_mode()
                    elif ch in ('r', 'R'):
                        self.handle_stop_collect()
                    elif ch in ('d', 'D'):
                        self.handle_discard_episode()
                    elif ch in ('q', 'Q'):
                        self.handle_quit()
                        break
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def handle_start_collect(self):
        """Handle start-collection key commands."""
        logger.warning('Start collecting (s pressed)')

        self.is_human_mode = False
        self.publish_global_state()

        self.control_cmd_pub.publish(String('start_collect'))

        for arm_name in self.arm_names:
            self.arm_mode_pubs[arm_name].publish(String('slave'))
            rospy.sleep(0.05)

        logger.info('Setting all arms to subscribe Topic1 '
                    '(inference mode)')
        self._set_arms_subscribe_inference()

        rospy.sleep(0.1)

        self.is_collecting = True
        self.is_episode_active = True
        self.publish_global_state()

        self.collector_cmd_pub.publish(String('start_episode'))
        logger.info('Sent start_episode command to collector')

        logger.info('Collection started in INFERENCE mode')
        logger.info('Press "h" to switch to HUMAN mode')

    def handle_human_mode(self):
        """Switch to human teleoperation mode."""
        logger.warning('Switch to HUMAN mode (h pressed)')

        self.is_human_mode = True
        self.publish_global_state()

        self.control_cmd_pub.publish(String('human_mode'))

        self.collector_cmd_pub.publish(String('set_human_mode'))

        logger.info('Setting rear arms to MASTER mode')
        for arm_name in self.rear_arm_names:
            self.arm_mode_pubs[arm_name].publish(String('master'))
            rospy.sleep(0.1)

        logger.info('Waiting for rear arms to '
                    'publish current positions...')
        rospy.sleep(0.5)

        logger.info('Setting front arms to subscribe Topic2 '
                    '(human mode)')
        self._set_front_arms_subscribe_human()

        rospy.sleep(0.1)

        logger.info('HUMAN mode activated')
        logger.info('Rear arms will publish to Topic2, '
                    'front arms will follow')

    def handle_inference_mode(self):
        """Switch to inference mode."""
        logger.warning('Switch to INFERENCE mode (i pressed)')

        self.is_human_mode = False
        self.publish_global_state()

        self.control_cmd_pub.publish(String('inference_mode'))

        self.collector_cmd_pub.publish(String('set_inference_mode'))

        logger.info('Setting all arms to SLAVE mode')
        for arm_name in self.arm_names:
            self.arm_mode_pubs[arm_name].publish(String('slave'))
            rospy.sleep(0.05)

        logger.info('Setting all arms to subscribe Topic1 '
                    '(inference mode)')
        self._set_arms_subscribe_inference()

        rospy.sleep(0.1)

        logger.info('INFERENCE mode activated')
        logger.info('All arms will follow Topic1 (model output)')

    def handle_stop_collect(self):
        """Stop collection, finish the episode, and home the arms."""
        logger.warning('Stop collecting (r pressed)')

        self.control_cmd_pub.publish(String('stop_collect'))

        logger.info('Repeatedly unsubscribing all arms '
                    'and setting SLAVE mode...')
        for i in range(3):
            logger.info(f'stop_collect mode/switch attempt {i+1}/3')
            for arm_name in self.arm_names:
                self.arm_subscribe_pubs[arm_name].publish(String(''))
            for arm_name in self.arm_names:
                self.arm_mode_pubs[arm_name].publish(String('slave'))
            rospy.sleep(0.2)

        self.is_collecting = False
        self.is_episode_active = False
        self.publish_global_state()

        logger.info('Moving all arms to home position '
                    '(stop_collect)...')
        for i in range(3):
            logger.info(f'stop_collect move_home attempt {i+1}/3')
            self.control_cmd_pub.publish(String('move_home'))
            rospy.sleep(2.0)

        logger.info('Collection stopped')
        sys.stdout.flush()

        self.collector_cmd_pub.publish(String('finish_episode'))
        logger.info('Sent finish_episode command to collector')
        sys.stdout.flush()

    def handle_discard_episode(self):
        """Discard the active episode and home the arms."""
        logger.warning('Discard episode (d pressed) - '
                       'data will be DELETED')

        self.control_cmd_pub.publish(String('stop_collect'))

        logger.info('Unsubscribing all arms and setting '
                    'SLAVE mode (discard)...')
        for i in range(3):
            for arm_name in self.arm_names:
                self.arm_subscribe_pubs[arm_name].publish(String(''))
            for arm_name in self.arm_names:
                self.arm_mode_pubs[arm_name].publish(String('slave'))
            rospy.sleep(0.2)

        self.is_collecting = False
        self.is_episode_active = False
        self.publish_global_state()

        logger.info('Moving all arms to home position (discard)...')
        for i in range(3):
            self.control_cmd_pub.publish(String('move_home'))
            rospy.sleep(2.0)

        logger.info('Episode discarded, sending discard command '
                    'to collector')
        sys.stdout.flush()

        self.collector_cmd_pub.publish(String('discard_episode'))
        logger.info('Sent discard_episode command to collector')
        sys.stdout.flush()

    def handle_quit(self):
        """Stop all work, home the arms, and shut down."""
        logger.warning('Quit key pressed, shutting down...')

        self.is_running = False

        self.is_collecting = False
        self.is_episode_active = False
        self.publish_global_state()

        logger.info('Unsubscribing all arms (quit)...')
        for i in range(3):
            for arm_name in self.arm_names:
                self.arm_subscribe_pubs[arm_name].publish(String(''))
            rospy.sleep(0.1)

        logger.info('Moving all arms to home position '
                    '(quit, end pose)...')
        for i in range(3):
            logger.info(f'quit move_home_endpose attempt {i+1}/3')
            self.control_cmd_pub.publish(String('move_home_endpose'))
            rospy.sleep(2.0)

        logger.info('Setting all arms to SLAVE mode and '
                    'broadcasting quit/shutdown...')
        for arm_name in self.arm_names:
            self.arm_mode_pubs[arm_name].publish(String('slave'))
        self.collector_cmd_pub.publish(String('shutdown'))
        self.control_cmd_pub.publish(String('quit'))
        rospy.sleep(0.5)

        self.shutdown()
        logger.info('Shutting down node...')
        rospy.signal_shutdown('User requested quit')

    def shutdown(self):
        """Clean up controller resources."""
        logger.info('Cleaning up resources...')
        pass


def main():
    """Initialize ROS and start the controller node."""
    rospy.init_node('dagger_controller', anonymous=False)

    try:
        controller = DaggerController()

        rospy.on_shutdown(controller.shutdown)

        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        logger.error('Fatal error: %s\n%s', e, traceback.format_exc())
    finally:
        logger.info('Process exiting.')
        sys.exit(0)


if __name__ == '__main__':
    main()
