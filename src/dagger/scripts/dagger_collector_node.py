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
"""dagger_collector_node.py.

Collector node that consumes synchronized observations, action topics, and
collection commands, then writes parquet episodes.
"""

from __future__ import annotations
import json
import os
import threading
import time
import traceback
import uuid
from collections import deque

import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, String
from trajectory_msgs.msg import JointTrajectory

from dagger.collectors.sync_frame_collector import SyncFrameDataCollector
from dagger.infra.dagger_log import get_logger
from dagger.runtime_config import (as_str, get_ros_override, get_ros_param,
                                   load_runtime_config_from_ros)
from dagger_msgs.msg import (EpisodeInfo, GlobalState, InputRequest,
                             InputResult, SyncedObservation)

logger = get_logger(__name__, prefix='DaggerCollector')

INPUT_WAIT_POLL_INTERVAL = 0.5
CAMERA_KEYS_BY_OBS_FIELD = {
    'cam_high': 'head',
    'cam_front': 'front',
    'cam_left_wrist': 'left_wrist',
    'cam_right_wrist': 'right_wrist',
}


class DaggerCollector:
    """Data collection node."""

    def __init__(self):
        self.runtime_config = load_runtime_config_from_ros('~')
        runtime = self.runtime_config.runtime
        collector = self.runtime_config.collector
        topics = self.runtime_config.topics

        self.ckpt_dir = get_ros_param('ckpt_dir', collector.ckpt_dir)
        self.task_id = get_ros_param('task_id', collector.task_id)
        self.scene_dir = get_ros_param('scene_dir', collector.scene_dir)
        self.save_data_base_dir = get_ros_param(
            'save_data_base_dir',
            as_str(self.runtime_config.paths.data_buffer_dir),
        )
        self.data_collection_rate = get_ros_override(
            'data_collection_rate',
            runtime.data_collection_rate,
            value_type=float,
        )
        self.debug = get_ros_override('debug', runtime.debug, value_type=bool)
        # Reject raw action chunks whose stamp is too far from sync_time.
        self.raw_action_chunk_max_sync_diff = max(
            0.0,
            get_ros_override(
                'raw_action_chunk_max_sync_diff',
                0.5,
                value_type=float,
            ),
        )

        self.is_collecting = False
        self.is_human_mode = False
        self.is_episode_active = False
        self.is_episode_started = False
        self.is_running = True

        self.saved_frame_count = 0
        self.skipped_incomplete_frame_count = 0
        self.current_episode_id = 0
        self.synced_obs_queue_size = max(
            1,
            get_ros_override(
                'synced_observation_queue_size',
                runtime.synced_observation_queue_size,
                value_type=int,
            ),
        )
        self.synced_observation_drain_timeout = max(
            0.0,
            get_ros_override(
                'synced_observation_drain_timeout',
                runtime.synced_observation_drain_timeout,
                value_type=float,
            ),
        )
        self.dropped_synced_obs_count = 0
        self.received_synced_obs_count = 0
        self._last_synced_obs_queue_warn_time = 0.0
        self.synced_obs_save_in_progress = 0

        self.camera_names = self._resolve_camera_names()

        self.puppet_arm_left_cmd_deque = deque(maxlen=1000)
        self.puppet_arm_right_cmd_deque = deque(maxlen=1000)
        self.puppet_arm_left_raw_action_chunk_deque = deque(maxlen=1000)
        self.puppet_arm_right_raw_action_chunk_deque = deque(maxlen=1000)

        cmd_left_topic = get_ros_param('puppet_arm_left_cmd_topic',
                                       topics.master_joint_left)
        cmd_right_topic = get_ros_param('puppet_arm_right_cmd_topic',
                                        topics.master_joint_right)
        raw_left_topic = get_ros_param('puppet_arm_left_raw_action_topic',
                                       topics.raw_action_left)
        raw_right_topic = get_ros_param('puppet_arm_right_raw_action_topic',
                                        topics.raw_action_right)

        if cmd_left_topic:
            rospy.Subscriber(
                cmd_left_topic,
                JointState,
                lambda msg: self.puppet_arm_left_cmd_deque.append(msg),
                queue_size=1000,
                tcp_nodelay=True)
        if cmd_right_topic:
            rospy.Subscriber(
                cmd_right_topic,
                JointState,
                lambda msg: self.puppet_arm_right_cmd_deque.append(msg),
                queue_size=1000,
                tcp_nodelay=True)
        if raw_left_topic:
            rospy.Subscriber(
                raw_left_topic,
                JointTrajectory,
                lambda msg: self.puppet_arm_left_raw_action_chunk_deque.append(
                    msg),
                queue_size=1000,
                tcp_nodelay=True)
        if raw_right_topic:
            rospy.Subscriber(
                raw_right_topic,
                JointTrajectory,
                lambda msg: self.puppet_arm_right_raw_action_chunk_deque.
                append(msg),
                queue_size=1000,
                tcp_nodelay=True)

        self.bridge = CvBridge()

        self.data_collector = None
        if not self.ckpt_dir:
            logger.error('ckpt_dir is required but not provided. '
                         'DataCollector will be DISABLED.')
        else:
            checkpoint_folder_name = os.path.basename(
                self.ckpt_dir.rstrip('/'))
            task_id = self.task_id if self.task_id else checkpoint_folder_name
            logger.info('Using SyncFrameDataCollector')
            self.data_collector = SyncFrameDataCollector(
                base_dir=self.save_data_base_dir,
                scene_dir=self.scene_dir,
                checkpoint_dir=self.ckpt_dir,
                task_id=task_id,
                default_camera_names=(
                    self.runtime_config.default_saved_camera_names()),
            )

        self.collector_cmd_sub = rospy.Subscriber(
            topics.dagger_collector_command,
            String,
            self.command_callback,
            queue_size=10)
        self.global_state_sub = rospy.Subscriber(
            topics.dagger_global_state,
            GlobalState,
            self.global_state_callback,
            queue_size=10)
        self.episode_result_pub = rospy.Publisher(
            topics.dagger_episode_result, Bool, queue_size=10)
        self.episode_info_pub = rospy.Publisher(
            topics.dagger_episode_info, EpisodeInfo, queue_size=10, latch=True)

        self.synced_obs_deque = deque(maxlen=self.synced_obs_queue_size)
        self.synced_obs_lock = threading.Lock()
        self.synced_obs_sub = rospy.Subscriber(
            topics.robot_observation_sync,
            SyncedObservation,
            self._synced_observation_callback,
            queue_size=self.synced_obs_queue_size,
        )

        self.input_request_pub = rospy.Publisher(
            topics.dagger_input_request, InputRequest, queue_size=10)
        self.input_result_sub = rospy.Subscriber(
            topics.dagger_input_result,
            InputResult,
            self._handle_input_result,
            queue_size=10)
        self.pending_input_requests = {}
        self.input_request_lock = threading.Lock()

        if self.data_collector is not None:
            self.data_thread = threading.Thread(
                target=self._data_collection_loop,
                daemon=True,
                name='data_collection_thread',
            )
            self.data_thread.start()

        logger.info(
            'Node started, synced_observation_queue_size=%d, '
            'synced_observation_drain_timeout=%.1fs',
            self.synced_obs_queue_size,
            self.synced_observation_drain_timeout,
        )

    def _resolve_camera_names(self):
        """Resolve enabled cameras from ROS topic configuration."""
        names = get_ros_override('camera_names', None, value_type=list)
        if names is not None:
            return names
        names = []
        camera_topics = self.runtime_config.camera_sync_topics()
        for key in self.runtime_config.camera_order:
            camera = self.runtime_config.cameras[key]
            topic = get_ros_param(camera.ros_param, camera_topics.get(key, ''))
            if topic:
                names.append(camera.name)
        return names or list(self.runtime_config.default_saved_camera_names())

    def shutdown(self):
        logger.info('Shutting down...')
        self.is_running = False

    def command_callback(self, msg):
        cmd = msg.data
        logger.info(f'Received command: {cmd}')
        if cmd == 'start_episode':
            self._handle_start_episode()
        elif cmd == 'finish_episode':
            self._handle_finish_episode()
        elif cmd == 'discard_episode':
            self._handle_discard_episode()
        elif cmd == 'set_human_mode':
            self.is_human_mode = True
            if self.data_collector is not None:
                self.data_collector.set_human_mode(True)
        elif cmd == 'set_inference_mode':
            self.is_human_mode = False
            if self.data_collector is not None:
                self.data_collector.set_human_mode(False)
        elif cmd == 'shutdown':
            logger.info('Received shutdown command')
            self.is_running = False
            rospy.signal_shutdown('Shutdown command from controller')

    def global_state_callback(self, msg):
        self.is_collecting = msg.is_collecting
        self.is_human_mode = msg.is_human_mode
        self.is_episode_active = msg.is_episode_active
        if self.data_collector is not None:
            self.data_collector.set_human_mode(self.is_human_mode)

    def _synced_observation_callback(self, msg: SyncedObservation) -> None:
        if not (self.is_episode_started and self.is_episode_active
                and self.is_collecting):
            return

        with self.synced_obs_lock:
            if len(self.synced_obs_deque) >= self.synced_obs_queue_size:
                self.dropped_synced_obs_count += 1
                now = rospy.Time.now().to_sec()
                if now - self._last_synced_obs_queue_warn_time >= 2.0:
                    self._last_synced_obs_queue_warn_time = now
                    logger.warning(
                        'Synced observation queue is full '
                        '(size=%d, dropped_oldest=%d). '
                        'Increase synced_observation_queue_size or reduce '
                        'frame save latency.',
                        self.synced_obs_queue_size,
                        self.dropped_synced_obs_count,
                    )

            self.synced_obs_deque.append(msg)
            self.received_synced_obs_count += 1

    def _handle_input_result(self, msg):
        with self.input_request_lock:
            if msg.request_id in self.pending_input_requests:
                event, container = self.pending_input_requests[msg.request_id]
                container['result'] = msg
                event.set()

    def request_user_input(
        self,
        prompt: str,
        valid_chars: str = '01',
        timeout: float = -1.0,
    ) -> tuple[bool, str, str]:
        """Request operator input through ``DaggerController``.

        The request is registered before publishing to avoid response races,
        and pending entries are always cleaned up on exit.

        Args:
            prompt: Prompt displayed by the controller.
            valid_chars: Accepted input characters.
            timeout: Wait timeout in seconds. Non-positive waits until a
                response, ROS shutdown, or node stop.

        Returns:
            ``(success, input_char, error_message)``.
        """
        request_id = str(uuid.uuid4())

        req = InputRequest()
        req.request_id = request_id
        req.prompt = prompt
        req.valid_chars = valid_chars
        req.timeout = timeout

        event = threading.Event()
        container = {'result': None}
        with self.input_request_lock:
            self.pending_input_requests[request_id] = (event, container)

        self.input_request_pub.publish(req)
        logger.info(f'Input request: {prompt}')

        triggered = False
        if timeout > 0:
            triggered = event.wait(timeout)
        else:
            while not rospy.is_shutdown() and self.is_running:
                if event.wait(INPUT_WAIT_POLL_INTERVAL):
                    triggered = True
                    break

        with self.input_request_lock:
            self.pending_input_requests.pop(request_id, None)
            result_msg = container['result']

        if result_msg is not None:
            return (result_msg.success, result_msg.input,
                    result_msg.error_message)
        if not triggered:
            if rospy.is_shutdown() or not self.is_running:
                return (False, '', 'Shutdown while waiting for input')
            return (False, '', 'Timeout waiting for input')
        return (False, '', 'Request was cancelled')

    def _handle_start_episode(self):
        if self.data_collector is None:
            logger.warning('DataCollector is not initialized')
            return
        if self.is_episode_started:
            logger.warning('Episode already started')
            return

        trajectory_type = 'human' if self.is_human_mode else 'rollout'
        logger.info(f'Starting episode (type={trajectory_type})')

        self.saved_frame_count = 0
        self.skipped_incomplete_frame_count = 0
        self.dropped_synced_obs_count = 0
        self.received_synced_obs_count = 0
        self._clear_synced_observation_queue()
        self._clear_action_message_queues()
        self.data_collector.start_new_episode(trajectory_type=trajectory_type)
        self.is_episode_started = True
        self.current_episode_id = getattr(self.data_collector, 'episode_id', 0)

        ep_msg = EpisodeInfo()
        ep_msg.episode_id = self.current_episode_id
        ep_msg.is_active = True
        ep_msg.trajectory_type = trajectory_type
        ep_msg.stamp = rospy.Time.now()
        self.episode_info_pub.publish(ep_msg)

        logger.info('Episode started')

    def _handle_finish_episode(self):
        if self.data_collector is None:
            logger.warning('DataCollector is not initialized')
            return
        if not self.is_episode_started:
            logger.warning('No episode to finish')
            return

        self.is_collecting = False
        self.is_episode_active = False
        if not self._wait_for_synced_observation_drain():
            self.is_episode_started = False
            self._clear_synced_observation_queue()
            self._wait_for_synced_observation_drain(timeout=2.0)

        success, input_str, error_msg = self.request_user_input(
            prompt='Episode Success? [1=success / 0=failure] : ',
            valid_chars='01',
            timeout=-1.0,
        )
        episode_success = (input_str == '1') if (
            success and input_str in ('0', '1')) else False
        if not success:
            logger.warning(f'Input failed: {error_msg}, default=failure')

        score = None
        if episode_success:
            for attempt in range(10):
                ok, val, err = self.request_user_input(
                    prompt='Episode Score (1-5): ',
                    valid_chars='12345',
                    timeout=-1.0,
                )
                if ok and val in ('1', '2', '3', '4', '5'):
                    score = int(val)
                    break
                logger.warning(f'Invalid score (retry {attempt + 1}/10)')

        trajectory_type = 'human' if self.is_human_mode else 'rollout'

        self.data_collector.finish_episode(
            success=episode_success,
            score=score,
            trajectory_type=trajectory_type,
            camera_names=self.camera_names,
        )
        self.is_episode_started = False
        self._clear_synced_observation_queue()

        ep_msg = EpisodeInfo()
        ep_msg.episode_id = self.current_episode_id
        ep_msg.is_active = False
        ep_msg.trajectory_type = trajectory_type
        ep_msg.stamp = rospy.Time.now()
        self.episode_info_pub.publish(ep_msg)

        self.episode_result_pub.publish(Bool(episode_success))
        logger.info(f'Episode finished (success={episode_success})')
        self._log_collection_summary('Episode finished')

    def _handle_discard_episode(self):
        """Discard the active episode and delete any saved data."""
        if self.data_collector is None:
            logger.warning('DataCollector is not initialized')
            return
        if not self.is_episode_started:
            logger.warning('No episode to discard')
            return

        logger.warning('Discarding current episode and deleting all data')

        self.is_collecting = False
        self.is_episode_active = False
        self.is_episode_started = False
        self._clear_synced_observation_queue()
        self._clear_action_message_queues()
        self._wait_for_synced_observation_drain(timeout=2.0)
        self.data_collector.discard_episode()

        ep_msg = EpisodeInfo()
        ep_msg.episode_id = self.current_episode_id
        ep_msg.is_active = False
        ep_msg.trajectory_type = 'discarded'
        ep_msg.stamp = rospy.Time.now()
        self.episode_info_pub.publish(ep_msg)

        logger.info('Episode discarded successfully')
        self._log_collection_summary('Episode discarded')

    def _data_collection_loop(self):
        """Consume synced observations and write parquet frames."""
        rate = rospy.Rate(self.data_collection_rate)

        while not rospy.is_shutdown() and self.is_running:
            if self.data_collector is None:
                rospy.sleep(0.1)
                continue

            if not self.is_episode_started:
                self._clear_synced_observation_queue()
                rate.sleep()
                continue

            if not (self.is_episode_active and self.is_collecting):
                if self._synced_observation_queue_len() == 0:
                    rate.sleep()
                    continue

            obs = self._pop_synced_observation()
            if obs is None:
                rate.sleep()
                continue

            try:
                self._save_frame_from_synced(obs)
            except Exception as e:
                logger.error('Save frame failed: %s\n%s', e,
                             traceback.format_exc())
            finally:
                self._mark_synced_observation_saved()

            if self._synced_observation_queue_len() == 0:
                rate.sleep()

    def _pop_synced_observation(self) -> SyncedObservation | None:
        """Pop the oldest queued synced observation, preserving frame order."""
        with self.synced_obs_lock:
            if len(self.synced_obs_deque) == 0:
                return None
            obs = self.synced_obs_deque.popleft()
            self.synced_obs_save_in_progress += 1
            return obs

    def _mark_synced_observation_saved(self) -> None:
        """Mark one popped synced observation as fully handled."""
        with self.synced_obs_lock:
            self.synced_obs_save_in_progress = max(
                0,
                self.synced_obs_save_in_progress - 1,
            )

    def _clear_synced_observation_queue(self) -> None:
        """Drop stale synced observations outside an active episode."""
        with self.synced_obs_lock:
            self.synced_obs_deque.clear()

    def _clear_action_message_queues(self) -> None:
        """Drop command messages buffered before the current episode."""
        self.puppet_arm_left_cmd_deque.clear()
        self.puppet_arm_right_cmd_deque.clear()
        self.puppet_arm_left_raw_action_chunk_deque.clear()
        self.puppet_arm_right_raw_action_chunk_deque.clear()

    def _synced_observation_queue_len(self) -> int:
        """Return the number of queued synced observations."""
        with self.synced_obs_lock:
            return len(self.synced_obs_deque)

    def _synced_observation_pending_count(self) -> int:
        """Return queued plus in-progress synced observations."""
        with self.synced_obs_lock:
            return (len(self.synced_obs_deque) +
                    self.synced_obs_save_in_progress)

    def _wait_for_synced_observation_drain(
        self,
        timeout: float | None = None,
    ) -> bool:
        """Wait for queued synced observations before finalizing."""
        if timeout is None:
            timeout = self.synced_observation_drain_timeout
        deadline = time.monotonic() + timeout
        while not rospy.is_shutdown() and self.is_running:
            pending = self._synced_observation_pending_count()
            if pending == 0:
                return True
            if time.monotonic() >= deadline:
                logger.warning(
                    'Timed out waiting for synced observation queue to drain '
                    '(pending=%d, timeout=%.1fs)',
                    pending,
                    timeout,
                )
                return False
            rospy.sleep(0.02)
        return False

    def _log_collection_summary(self, prefix: str) -> None:
        """Log frame queue counters for diagnosing collection drops."""
        logger.info(
            '%s summary: received_synced=%d, saved=%d, '
            'skipped_incomplete=%d, dropped_synced=%d, pending=%d',
            prefix,
            self.received_synced_obs_count,
            self.saved_frame_count,
            self.skipped_incomplete_frame_count,
            self.dropped_synced_obs_count,
            self._synced_observation_pending_count(),
        )

    def _save_frame_from_synced(self, obs: SyncedObservation) -> None:
        """Save one synced observation with aligned local action data.

        Action and raw-action queues consume all messages with
        ``ts <= frame_time`` and keep the latest message.
        """
        timestamps = {}
        if obs.timestamps_json:
            try:
                timestamps = json.loads(obs.timestamps_json)
            except (json.JSONDecodeError, TypeError) as exc:
                logger.warning(
                    'Failed to parse timestamps_json; '
                    'falling back to an empty dict: %s', exc)
        if 'sync_time' not in timestamps:
            timestamps['sync_time'] = obs.stamp.to_sec() if obs.stamp else 0.0

        frame_time = timestamps.get('sync_time', 0.0)

        puppet_arm_left_cmd = self._pop_up_to(self.puppet_arm_left_cmd_deque,
                                              frame_time)
        puppet_arm_right_cmd = self._pop_up_to(self.puppet_arm_right_cmd_deque,
                                               frame_time)

        action = self._build_action(puppet_arm_left_cmd, puppet_arm_right_cmd)

        if puppet_arm_left_cmd is not None:
            timestamps['puppet_arm_left_cmd'] = (
                puppet_arm_left_cmd.header.stamp.to_sec())
        if puppet_arm_right_cmd is not None:
            timestamps['puppet_arm_right_cmd'] = (
                puppet_arm_right_cmd.header.stamp.to_sec())

        left_chunk = self._pop_up_to(
            self.puppet_arm_left_raw_action_chunk_deque, frame_time)
        right_chunk = self._pop_up_to(
            self.puppet_arm_right_raw_action_chunk_deque, frame_time)

        if self._should_record_raw_action_chunk(left_chunk, right_chunk,
                                                frame_time):
            if left_chunk is not None:
                timestamps['puppet_arm_left_raw_action_chunk'] = (
                    left_chunk.header.stamp.to_sec())
            if right_chunk is not None:
                timestamps['puppet_arm_right_raw_action_chunk'] = (
                    right_chunk.header.stamp.to_sec())

            current_frame_id = getattr(self.data_collector, 'step_id', None)
            self.data_collector.record_raw_action_chunk(
                left_chunk,
                right_chunk,
                frame_id=current_frame_id,
            )
        elif left_chunk is not None or right_chunk is not None:
            chunk_ts = self._raw_action_chunk_timestamp(
                left_chunk, right_chunk)
            logger.debug(
                'Skipping stale raw action chunk at frame_time=%.3f '
                '(chunk_ts=%.3f, diff=%.3fs, max=%.3fs)',
                frame_time,
                chunk_ts,
                abs(chunk_ts - frame_time),
                self.raw_action_chunk_max_sync_diff,
            )

        self.saved_frame_count += 1
        if self.debug and self.saved_frame_count % 30 == 0:
            logger.info(
                '[DaggerCollector][DEBUG] Frame #%d (human=%s, qpos_dim=%d)',
                self.saved_frame_count,
                self.is_human_mode,
                len(obs.qpos) if obs.qpos else 0,
            )

        images = {}
        for obs_field, camera_key in CAMERA_KEYS_BY_OBS_FIELD.items():
            msg = getattr(obs, obs_field)
            if msg.data:
                camera_name = self.runtime_config.camera_name(camera_key)
                images[camera_name] = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

        if not self._has_complete_robot_state(obs):
            self.skipped_incomplete_frame_count += 1
            if self.skipped_incomplete_frame_count % 30 == 1:
                logger.warning(
                    'Skipping incomplete synced observation '
                    '(qpos=%d, qvel=%d, effort=%d, eepose=%d, skipped=%d)',
                    len(obs.qpos),
                    len(obs.qvel),
                    len(obs.effort),
                    len(obs.eepose),
                    self.skipped_incomplete_frame_count,
                )
            return

        qpos = np.asarray(obs.qpos, dtype=np.float32)
        qvel = np.asarray(obs.qvel, dtype=np.float32)
        effort = np.asarray(obs.effort, dtype=np.float32)
        eepose = np.asarray(obs.eepose, dtype=np.float32)

        self.data_collector.save_frame_from_synced(
            qpos=qpos,
            qvel=qvel,
            effort=effort,
            eepose=eepose,
            action=action,
            images=images,
            timestamps=timestamps,
            camera_names=self.camera_names,
        )

    @staticmethod
    def _has_complete_robot_state(obs: SyncedObservation) -> bool:
        """Require both arms and both end poses before saving a frame."""
        return (len(obs.qpos) == 14 and len(obs.qvel) == 14
                and len(obs.effort) == 14 and len(obs.eepose) == 14)

    @staticmethod
    def _pop_up_to(dq, frame_time: float):
        """Pop all messages with ``ts <= frame_time`` and return the latest."""
        latest = None
        while len(dq) > 0 and dq[0].header.stamp.to_sec() <= frame_time:
            latest = dq.popleft()
        return latest

    @staticmethod
    def _raw_action_chunk_timestamp(left_chunk, right_chunk) -> float:
        """Return the header stamp used for raw action chunk alignment."""
        if left_chunk is not None and hasattr(left_chunk, 'header'):
            return left_chunk.header.stamp.to_sec()
        if right_chunk is not None and hasattr(right_chunk, 'header'):
            return right_chunk.header.stamp.to_sec()
        return 0.0

    def _should_record_raw_action_chunk(self, left_chunk, right_chunk,
                                        frame_time: float) -> bool:
        """Only persist chunks that are time-aligned with the synced frame."""
        if left_chunk is None and right_chunk is None:
            return False
        chunk_ts = self._raw_action_chunk_timestamp(left_chunk, right_chunk)
        return (abs(chunk_ts - frame_time) <=
                self.raw_action_chunk_max_sync_diff)

    @staticmethod
    def _build_action(left_cmd, right_cmd) -> np.ndarray:
        """Build a 14D action vector from left/right command messages."""
        left = np.asarray(left_cmd.position,
                          np.float32) if left_cmd is not None else np.zeros(
                              7, np.float32)
        right = np.asarray(right_cmd.position,
                           np.float32) if right_cmd is not None else np.zeros(
                               7, np.float32)
        return np.concatenate((left, right), axis=0)


def main():
    rospy.init_node('dagger_collector', anonymous=False)
    try:
        collector = DaggerCollector()
        rospy.on_shutdown(collector.shutdown)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        logger.error('Fatal error: %s\n%s', e, traceback.format_exc())


if __name__ == '__main__':
    main()
