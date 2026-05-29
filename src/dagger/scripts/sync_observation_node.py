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
"""Independent observation sync node for publishing ``SyncedObservation``."""

from __future__ import annotations
import json
import os
import time
from typing import Any, Dict, List

import numpy as np
import rospy
from cv_bridge import CvBridge
from std_msgs.msg import String

from dagger.infra.dagger_log import get_logger
from dagger.infra.frame_sync import FrameData
from dagger.infra.ros_observation_buffer import (RosObservationBuffer,
                                                 RosObservationBufferConfig)
from dagger.runtime_config import (as_str, get_ros_override, get_ros_param,
                                   load_runtime_config_from_ros)
from dagger_msgs.msg import EpisodeInfo, SyncedObservation

logger = get_logger(__name__, prefix='SyncObservationNode')


def _pose_to_vec7(ps) -> np.ndarray:
    """Convert ``PoseStamped`` to ``[x, y, z, qx, qy, qz, qw]``."""
    p = ps.pose.position
    q = ps.pose.orientation
    return np.array(
        [p.x, p.y, p.z, q.x, q.y, q.z, q.w],
        dtype=np.float32,
    )


class SyncObservationNode:
    """Periodically sync raw ROS topics through ``RosObservationBuffer``."""

    def __init__(self) -> None:
        self.runtime_config = load_runtime_config_from_ros('~')
        runtime = self.runtime_config.runtime
        topics = self.runtime_config.topics
        self.sync_observation_rate = get_ros_override(
            'sync_observation_rate',
            runtime.sync_observation_rate,
            value_type=float,
        )

        config = RosObservationBufferConfig.from_ros_params(
            '~', self.runtime_config)
        self.debug = config.debug
        self.observation_buffer = RosObservationBuffer(config)
        self.bridge = CvBridge()
        self.synced_observation_pub_queue_size = max(
            1,
            get_ros_override(
                'synced_observation_pub_queue_size',
                runtime.synced_observation_pub_queue_size,
                value_type=int,
            ),
        )

        self.current_episode_id: int = 0
        self.current_step_id: int = 0
        self.episode_active: bool = False

        rospy.Subscriber(
            topics.dagger_episode_info,
            EpisodeInfo,
            self._episode_callback,
            queue_size=10,
        )

        self.obs_pub = rospy.Publisher(
            topics.robot_observation_sync,
            SyncedObservation,
            queue_size=self.synced_observation_pub_queue_size,
        )

        self.task_id = get_ros_param(
            'task_id', self.runtime_config.collector.task_id or '')
        self.task_text = get_ros_param('task_text', runtime.task_text)

        self.is_running = True

        self._sync_failures: List[Dict] = []
        self._sync_fail_log_dir = get_ros_param(
            'sync_fail_log_dir',
            as_str(self.runtime_config.paths.sync_fail_log_dir),
        )
        self._sync_fail_counts: Dict[str, int] = {}
        self._last_fail_log_time: float = 0.0
        self._fail_log_interval: float = 2.0

        rospy.on_shutdown(self.shutdown)

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

        logger.info(
            'Initialized, rate=%.1f Hz, synced_observation_pub_queue_size=%d',
            self.sync_observation_rate,
            self.synced_observation_pub_queue_size,
        )

    def _episode_callback(self, msg: EpisodeInfo) -> None:
        """Align episode id and step counter from ``EpisodeInfo``."""
        if msg.is_active:
            self.current_episode_id = int(msg.episode_id)
            self.current_step_id = 0
            self.episode_active = True
            logger.info(
                'Episode %d started '
                '(trajectory_type=%s)',
                self.current_episode_id,
                msg.trajectory_type,
            )
        else:
            if (self.episode_active
                    and msg.episode_id == self.current_episode_id):
                self.episode_active = False
                logger.info('Episode %d finished', self.current_episode_id)

    def _control_command_callback(self, msg: String) -> None:
        if msg.data == 'quit':
            logger.info('Received quit command, shutting down')
            self.is_running = False
            rospy.signal_shutdown('Quit command from controller')

    def _collector_command_callback(self, msg: String) -> None:
        if msg.data == 'shutdown':
            logger.info(''
                        'Received shutdown command, shutting down')
            self.is_running = False
            rospy.signal_shutdown('Shutdown command from controller')

    def shutdown(self) -> None:
        """Stop accepting data and save sync failure records."""
        logger.info('Shutting down...')
        self.is_running = False
        with self.observation_buffer.accept_new_data_lock:
            self.observation_buffer.accept_new_data = False
        self._save_sync_failures()

    def _save_sync_failures(self) -> None:
        """Save sync failure records to a JSON file."""
        if not self._sync_failures:
            logger.info('No sync failures to save.')
            return

        os.makedirs(self._sync_fail_log_dir, exist_ok=True)
        ts_str = time.strftime('%Y%m%d_%H%M%S')
        filename = f'sync_failures_{ts_str}.json'
        filepath = os.path.join(self._sync_fail_log_dir, filename)

        summary: Dict[str, Any] = {
            'total_failures': len(self._sync_failures),
            'counts_by_reason': dict(self._sync_fail_counts),
            'records': self._sync_failures,
        }
        try:
            with open(filepath, 'w') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            logger.info('Saved %d sync failure records to %s',
                        len(self._sync_failures), filepath)
        except Exception as e:
            logger.error('Failed to save sync failures: %s', e)

    def spin(self) -> None:
        """Poll the observation buffer and publish synced observations."""
        rate = rospy.Rate(self.sync_observation_rate)

        with self.observation_buffer.accept_new_data_lock:
            self.observation_buffer.accept_new_data = True

        while not rospy.is_shutdown() and self.is_running:
            success, data = self.observation_buffer.get_frame()
            if not success:
                if isinstance(data, str) and data.startswith('empty_queues:'):
                    rate.sleep()
                    continue
                self._handle_sync_failure(data)
                rate.sleep()
                continue

            self._publish_observation(data)
            rate.sleep()

    def _handle_sync_failure(self, data: Any) -> None:
        """Record sync failures and throttle summary logs."""
        now = time.monotonic()

        if isinstance(data, dict):
            reason = data.get('reason', 'unknown')
            self._sync_failures.append(data)
            self._sync_fail_counts[reason] = self._sync_fail_counts.get(
                reason, 0) + 1

            if now - self._last_fail_log_time >= self._fail_log_interval:
                self._last_fail_log_time = now
                if reason == 'span_exceed_slop':
                    logger.warning(
                        'span_exceed_slop: '
                        'span=%.4fs slop=%.4fs | '
                        'anchor(frame_time)=%.6f from=[%s] | '
                        'max_at=[%s] ts=%.6f | '
                        'aligned: %s | '
                        'total_failures: %s',
                        data.get('span', 0),
                        data.get('sync_slop', 0),
                        data.get('frame_time', 0),
                        ','.join(data.get('anchor_cams', [])),
                        ','.join(data.get('max_at', [])),
                        data.get('frame_time_max', 0),
                        ' '.join(
                            f'{k}={v:.6f}'
                            for k, v in data.get('aligned_ts', {}).items()),
                        dict(self._sync_fail_counts),
                    )
                elif reason == 'source_lag':
                    logger.warning(
                        'source_lag: '
                        '%s lagging %.4fs behind frame_time=%.6f '
                        '(source_latest=%.6f) | '
                        'total_failures: %s',
                        data.get('lagging_source', '?'),
                        data.get('lag', 0),
                        data.get('frame_time', 0),
                        data.get('source_latest_ts', 0),
                        dict(self._sync_fail_counts),
                    )
                else:
                    logger.warning('sync fail: %s | total: %s', data,
                                   dict(self._sync_fail_counts))
        elif isinstance(data, str):
            self._sync_failures.append({
                'reason': data,
                'wall_time': rospy.Time.now().to_sec(),
            })
            self._sync_fail_counts[data] = self._sync_fail_counts.get(data,
                                                                      0) + 1
            if now - self._last_fail_log_time >= self._fail_log_interval:
                self._last_fail_log_time = now
                logger.warning('get_frame failed: %s | total: %s', data,
                               dict(self._sync_fail_counts))

    def _publish_observation(self, frame: FrameData) -> None:
        """Pack ``FrameData`` into ``SyncedObservation`` and publish it."""
        msg = SyncedObservation()
        msg.episode_id = int(self.current_episode_id)
        msg.step_id = int(self.current_step_id)
        msg.is_terminal = False

        sync_time = frame.timestamps.get('sync_time', 0.0) or 0.0
        msg.stamp = rospy.Time.from_sec(sync_time)

        if frame.img_head is not None:
            msg.cam_high = self.bridge.cv2_to_imgmsg(
                frame.img_head, encoding='bgr8')
        if frame.img_front is not None:
            msg.cam_front = self.bridge.cv2_to_imgmsg(
                frame.img_front, encoding='bgr8')
        if frame.img_left is not None:
            msg.cam_left_wrist = self.bridge.cv2_to_imgmsg(
                frame.img_left, encoding='bgr8')
        if frame.img_right is not None:
            msg.cam_right_wrist = self.bridge.cv2_to_imgmsg(
                frame.img_right, encoding='bgr8')

        left = frame.puppet_arm_left
        right = frame.puppet_arm_right
        if left is not None and right is not None:
            msg.qpos = np.concatenate(
                (
                    np.asarray(left.position, dtype=np.float32),
                    np.asarray(right.position, dtype=np.float32),
                ),
                axis=0,
            ).tolist()
            msg.qvel = np.concatenate(
                (
                    np.asarray(left.velocity, dtype=np.float32),
                    np.asarray(right.velocity, dtype=np.float32),
                ),
                axis=0,
            ).tolist()
            msg.effort = np.concatenate(
                (
                    np.asarray(left.effort, dtype=np.float32),
                    np.asarray(right.effort, dtype=np.float32),
                ),
                axis=0,
            ).tolist()

        ee_left = frame.puppet_ee_pose_left
        ee_right = frame.puppet_ee_pose_right
        if ee_left is not None and ee_right is not None:
            msg.eepose = np.concatenate(
                (_pose_to_vec7(ee_left), _pose_to_vec7(ee_right)),
                axis=0,
            ).tolist()

        msg.task_id = self.task_id
        msg.task_text = self.task_text

        try:
            msg.timestamps_json = json.dumps(frame.timestamps)
        except (TypeError, ValueError) as exc:
            logger.warning_throttle(
                5.0,
                'Failed to serialize timestamps_json; '
                'falling back to an empty string: %s',
                exc,
            )
            msg.timestamps_json = ''

        self.obs_pub.publish(msg)

        if self.episode_active:
            self.current_step_id += 1


def main() -> None:
    rospy.init_node('sync_observation_node', anonymous=False)
    node = SyncObservationNode()
    node.spin()


if __name__ == '__main__':
    main()
