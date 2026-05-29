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
"""Online Qwen3-VL reward inference ROS node."""

from __future__ import annotations
import time
import traceback

import rospy
from std_msgs.msg import String

from dagger.infra.async_worker import InferenceWorker
from dagger.infra.dagger_log import get_logger
from dagger.reward_models.qwen3_reward import Qwen3RewardModel
from dagger.runtime_config import (as_str, get_ros_override, get_ros_param,
                                   load_runtime_config_from_ros)
from dagger_msgs.msg import RewardSignal, SyncedObservation

logger = get_logger(__name__, prefix='Qwen3RewardNode')


class Qwen3RewardOnlineNode:
    """Online Qwen3-VL reward inference node."""

    def __init__(self) -> None:
        self.runtime_config = load_runtime_config_from_ros('~')
        topics = self.runtime_config.topics
        qwen3 = self.runtime_config.reward.qwen3

        model_path = get_ros_param('model_path', as_str(qwen3.model_path))
        if not model_path:
            logger.error('~model_path is not set; cannot load model')
            raise RuntimeError('model_path is required')

        prompt = get_ros_param('prompt', qwen3.prompt)
        f_interval = get_ros_override(
            'f_interval', qwen3.f_interval, value_type=int)
        t_interval = get_ros_override(
            't_interval', qwen3.t_interval, value_type=int)
        max_new_tokens = get_ros_override(
            'max_new_tokens', qwen3.max_new_tokens, value_type=int)
        self._source_tag = get_ros_param('source_tag', qwen3.source_tag)

        logger.info('Loading Qwen3-VL model from %s ...', model_path)
        self.reward_model = Qwen3RewardModel()
        self.reward_model.setup({
            'model_path':
            model_path,
            'prompt':
            prompt,
            'f_interval':
            f_interval,
            't_interval':
            t_interval,
            'max_new_tokens':
            max_new_tokens,
            'source_tag':
            self._source_tag,
            'camera_fields': (
                self.runtime_config.cameras['head'].msg_field,
                self.runtime_config.cameras['left_wrist'].msg_field,
                self.runtime_config.cameras['right_wrist'].msg_field,
            ),
        })
        logger.info('Model loaded successfully.')

        self.reward_pub = rospy.Publisher(
            topics.robot_reward_signal,
            RewardSignal,
            queue_size=10,
        )

        self._worker = InferenceWorker(
            self._run_inference,
            name='qwen3_reward_worker',
            queue_size=1,
        )
        self._worker.set_result_handler(self._publish_reward)
        self._worker.start()

        rospy.Subscriber(
            topics.robot_observation_sync,
            SyncedObservation,
            self._obs_callback,
            queue_size=1,
            tcp_nodelay=True,
        )

        rospy.Subscriber(
            topics.dagger_control_command,
            String,
            self._control_command_callback,
            queue_size=10,
        )
        rospy.Subscriber(
            topics.dagger_collector_command,
            String,
            self._collector_command_callback,
            queue_size=10,
        )

        rospy.on_shutdown(self._worker.stop)
        logger.info('Node initialized.')

    def _control_command_callback(self, msg: String) -> None:
        if msg.data == 'quit':
            logger.info('Received quit command, shutting down')
            self._worker.stop()
            rospy.signal_shutdown('Quit command from controller')

    def _collector_command_callback(self, msg: String) -> None:
        if msg.data == 'shutdown':
            logger.info('Received shutdown command, shutting down')
            self._worker.stop()
            rospy.signal_shutdown('Shutdown command from controller')

    def _obs_callback(self, msg: SyncedObservation) -> None:
        """Queue observations; inference runs in the worker thread."""
        self._worker.submit(msg)

    def _run_inference(self, msg: SyncedObservation) -> dict:
        start_t = time.time()
        result = self.reward_model.predict(msg)
        if result is None:
            return None
        result['__elapsed'] = time.time() - start_t
        result['__episode_id'] = int(msg.episode_id)
        result['__step_id'] = int(msg.step_id)
        result['__stamp'] = msg.stamp
        return result

    def _publish_reward(self, _msg, result: dict) -> None:
        logger.debug(
            'Inference: status=%s, value=%.3f, time=%.3fs',
            result.get('status_text', '?'),
            result['value'],
            result.get('__elapsed', 0.0),
        )

        out = RewardSignal()
        out.episode_id = result['__episode_id']
        out.step_id = result['__step_id']
        out.stamp = result['__stamp']
        out.value = float(result['value'])
        out.done = bool(result['done'])
        out.confidence = float(result['confidence'])
        out.aux_values = result.get('aux_values', [])
        out.source = self._source_tag

        self.reward_pub.publish(out)


def main() -> None:
    rospy.init_node('qwen3_reward_online_node', anonymous=False)
    try:
        Qwen3RewardOnlineNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:  # noqa: BLE001 - top-level log
        logger.error('Fatal error: %s\n%s', e, traceback.format_exc())


if __name__ == '__main__':
    main()
