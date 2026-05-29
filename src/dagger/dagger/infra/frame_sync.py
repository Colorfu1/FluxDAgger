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
"""Multi-sensor ROS frame synchronization.

The sync anchor is the minimum of the latest timestamps from enabled image
streams. Successful frames are consumed at ``frame_time`` and packed as
``FrameData``.

Image streams provide the sync anchor; arm state and end-pose streams are
aligned to that anchor.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Dict, NamedTuple, Optional

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory

if TYPE_CHECKING:
    from dagger.infra.ros_observation_buffer import RosObservationBuffer

__all__ = ['FrameData', 'sync_get_frame']


class FrameData(NamedTuple):
    """Packed synchronized frame returned by the sync pipeline."""

    img_head: Optional[np.ndarray]
    img_front: Optional[np.ndarray]
    img_left: Optional[np.ndarray]
    img_right: Optional[np.ndarray]
    puppet_arm_left: Optional[JointState]
    puppet_arm_right: Optional[JointState]
    puppet_ee_pose_left: Optional[PoseStamped]
    puppet_ee_pose_right: Optional[PoseStamped]
    puppet_arm_left_cmd: Optional[JointState]
    puppet_arm_right_cmd: Optional[JointState]
    puppet_arm_left_raw_action_chunk: Optional[JointTrajectory]
    puppet_arm_right_raw_action_chunk: Optional[JointTrajectory]
    timestamps: Dict[str, float]


SyncResult = tuple[bool, FrameData | str | dict]


def _build_lag_fail(source_name: str, latest_ts: float, frame_time: float,
                    img_latest_by_name) -> dict:
    """Build a structured diagnostic for a source lagging behind."""
    return {
        'reason': 'source_lag',
        'wall_time': rospy.Time.now().to_sec(),
        'lagging_source': source_name,
        'source_latest_ts': round(latest_ts, 6),
        'frame_time': round(frame_time, 6),
        'lag': round(frame_time - latest_ts, 6),
        'cam_latest': {n: round(ts, 6)
                       for n, ts in img_latest_by_name},
    }


def _first_ts_ge(deque_obj, t):
    for i in range(len(deque_obj)):
        if deque_obj[i].header.stamp.to_sec() >= t:
            return deque_obj[i].header.stamp.to_sec()
    return deque_obj[-1].header.stamp.to_sec() if len(deque_obj) > 0 else t


def sync_get_frame(op: 'RosObservationBuffer') -> SyncResult:
    """Return one synchronized frame from image and puppet-arm queues.

    Enabled image queues define ``frame_time``. Required puppet and end-pose
    queues must have data not older than ``frame_time``. Raw action chunks are
    consumed opportunistically and are not part of the strict sync anchor.

    Args:
        op: :class:`~dagger.infra.ros_observation_buffer.RosObservationBuffer`
            instance.

    Returns:
        ``(True, FrameData)`` on success, or ``(False, reason)`` on failure.
        ``reason`` may be a compact string or a structured diagnostic dict.
    """
    cfg = op.config
    use_front_camera = bool(cfg.img_front_topic)

    # Enabled image queues are required anchors for synchronization.
    if cfg.img_head_topic and len(op.img_head_deque) == 0:
        return (False, 'empty_queues: img_head_deque_empty')
    if cfg.img_left_topic and len(op.img_left_deque) == 0:
        return (False, 'empty_queues: img_left_deque_empty')
    if cfg.img_right_topic and len(op.img_right_deque) == 0:
        return (False, 'empty_queues: img_right_deque_empty')
    if use_front_camera and len(op.img_front_deque) == 0:
        return (False, 'empty_queues: img_front_deque_empty')

    # Use the minimum latest timestamp among enabled cameras as the frame time.
    img_latest_by_name = []
    if cfg.img_head_topic:
        img_latest_by_name.append(
            ('img_head', op.img_head_deque[-1].header.stamp.to_sec()))
    if cfg.img_left_topic:
        img_latest_by_name.append(
            ('img_left', op.img_left_deque[-1].header.stamp.to_sec()))
    if cfg.img_right_topic:
        img_latest_by_name.append(
            ('img_right', op.img_right_deque[-1].header.stamp.to_sec()))
    if use_front_camera:
        img_latest_by_name.append(
            ('img_front', op.img_front_deque[-1].header.stamp.to_sec()))

    if not img_latest_by_name:
        return (False, 'no_enabled_cameras')

    frame_time = min(ts for _, ts in img_latest_by_name)
    _ts_eps = 1e-6
    anchor_cams_min = [
        n for n, ts in img_latest_by_name if abs(ts - frame_time) < _ts_eps
    ]

    # Check required non-image queues before cleanup so popleft stays safe.
    if cfg.puppet_arm_left_topic:
        if len(op.puppet_arm_left_deque) == 0:
            return (False,
                    _build_lag_fail('puppet_arm_left', 0.0, frame_time,
                                    img_latest_by_name))
        _ts = op.puppet_arm_left_deque[-1].header.stamp.to_sec()
        if _ts < frame_time:
            return (False,
                    _build_lag_fail('puppet_arm_left', _ts, frame_time,
                                    img_latest_by_name))
    if cfg.puppet_arm_right_topic:
        if len(op.puppet_arm_right_deque) == 0:
            return (False,
                    _build_lag_fail('puppet_arm_right', 0.0, frame_time,
                                    img_latest_by_name))
        _ts = op.puppet_arm_right_deque[-1].header.stamp.to_sec()
        if _ts < frame_time:
            return (False,
                    _build_lag_fail('puppet_arm_right', _ts, frame_time,
                                    img_latest_by_name))
    if cfg.puppet_ee_pose_left_topic:
        if len(op.puppet_ee_pose_left_deque) == 0:
            return (False,
                    _build_lag_fail('puppet_ee_pose_left', 0.0, frame_time,
                                    img_latest_by_name))
        _ts = op.puppet_ee_pose_left_deque[-1].header.stamp.to_sec()
        if _ts < frame_time:
            return (False,
                    _build_lag_fail('puppet_ee_pose_left', _ts, frame_time,
                                    img_latest_by_name))
    if cfg.puppet_ee_pose_right_topic:
        if len(op.puppet_ee_pose_right_deque) == 0:
            return (False,
                    _build_lag_fail('puppet_ee_pose_right', 0.0, frame_time,
                                    img_latest_by_name))
        _ts = op.puppet_ee_pose_right_deque[-1].header.stamp.to_sec()
        if _ts < frame_time:
            return (False,
                    _build_lag_fail('puppet_ee_pose_right', _ts, frame_time,
                                    img_latest_by_name))
    # Compute span from the first message that will actually be consumed from
    # each queue, not from each queue tail. This keeps fast 200Hz arm queues
    # from inflating the frame span.
    # First timestamp at or after frame_time for every participating stream.
    aligned_ts = []
    if cfg.img_head_topic:
        aligned_ts.append(
            ('img_head', _first_ts_ge(op.img_head_deque, frame_time)))
    if cfg.img_left_topic:
        aligned_ts.append(
            ('img_left', _first_ts_ge(op.img_left_deque, frame_time)))
    if cfg.img_right_topic:
        aligned_ts.append(
            ('img_right', _first_ts_ge(op.img_right_deque, frame_time)))
    if use_front_camera:
        aligned_ts.append(
            ('img_front', _first_ts_ge(op.img_front_deque, frame_time)))
    if cfg.puppet_arm_left_topic and len(op.puppet_arm_left_deque) > 0:
        aligned_ts.append(('puppet_arm_left',
                           _first_ts_ge(op.puppet_arm_left_deque, frame_time)))
    if cfg.puppet_arm_right_topic and len(op.puppet_arm_right_deque) > 0:
        aligned_ts.append(('puppet_arm_right',
                           _first_ts_ge(op.puppet_arm_right_deque,
                                        frame_time)))
    if (cfg.puppet_ee_pose_left_topic
            and len(op.puppet_ee_pose_left_deque) > 0):
        aligned_ts.append(('puppet_ee_pose_left',
                           _first_ts_ge(op.puppet_ee_pose_left_deque,
                                        frame_time)))
    if (cfg.puppet_ee_pose_right_topic
            and len(op.puppet_ee_pose_right_deque) > 0):
        aligned_ts.append(('puppet_ee_pose_right',
                           _first_ts_ge(op.puppet_ee_pose_right_deque,
                                        frame_time)))
    frame_time_max = frame_time
    for _, ts in aligned_ts:
        frame_time_max = max(frame_time_max, ts)

    # Drop this frame when the participating stream span exceeds sync_slop.
    if abs(frame_time_max - frame_time) > op.sync_slop:
        # Match the success-path cleanup policy before returning.
        if cfg.img_head_topic:
            while (len(op.img_head_deque) > 0 and
                   op.img_head_deque[0].header.stamp.to_sec() <= frame_time):
                op.img_head_deque.popleft()
        if cfg.img_left_topic:
            while (len(op.img_left_deque) > 0 and
                   op.img_left_deque[0].header.stamp.to_sec() <= frame_time):
                op.img_left_deque.popleft()
        if cfg.img_right_topic:
            while (len(op.img_right_deque) > 0 and
                   op.img_right_deque[0].header.stamp.to_sec() <= frame_time):
                op.img_right_deque.popleft()
        if use_front_camera:
            while (len(op.img_front_deque) > 0 and
                   op.img_front_deque[0].header.stamp.to_sec() <= frame_time):
                op.img_front_deque.popleft()

        if cfg.puppet_arm_left_topic:
            while (len(op.puppet_arm_left_deque) > 0
                   and op.puppet_arm_left_deque[0].header.stamp.to_sec() <=
                   frame_time):
                op.puppet_arm_left_deque.popleft()
        if cfg.puppet_arm_right_topic:
            while (len(op.puppet_arm_right_deque) > 0
                   and op.puppet_arm_right_deque[0].header.stamp.to_sec() <=
                   frame_time):
                op.puppet_arm_right_deque.popleft()
        if cfg.puppet_ee_pose_left_topic:
            while (len(op.puppet_ee_pose_left_deque) > 0
                   and op.puppet_ee_pose_left_deque[0].header.stamp.to_sec() <=
                   frame_time):
                op.puppet_ee_pose_left_deque.popleft()
        if cfg.puppet_ee_pose_right_topic:
            while (len(op.puppet_ee_pose_right_deque) > 0
                   and op.puppet_ee_pose_right_deque[0].header.stamp.to_sec()
                   <= frame_time):
                op.puppet_ee_pose_right_deque.popleft()

        if cfg.puppet_arm_left_cmd_topic:
            while (len(op.puppet_arm_left_cmd_deque) > 0
                   and op.puppet_arm_left_cmd_deque[0].header.stamp.to_sec() <=
                   frame_time):
                op.puppet_arm_left_cmd_deque.popleft()
        if cfg.puppet_arm_right_cmd_topic:
            while (len(op.puppet_arm_right_cmd_deque) > 0
                   and op.puppet_arm_right_cmd_deque[0].header.stamp.to_sec()
                   <= frame_time):
                op.puppet_arm_right_cmd_deque.popleft()
        if cfg.puppet_arm_left_raw_action_topic:
            while (len(op.puppet_arm_left_raw_action_chunk_deque) > 0
                   and op.puppet_arm_left_raw_action_chunk_deque[0].header.
                   stamp.to_sec() <= frame_time):
                op.puppet_arm_left_raw_action_chunk_deque.popleft()
        if cfg.puppet_arm_right_raw_action_topic:
            while (len(op.puppet_arm_right_raw_action_chunk_deque) > 0
                   and op.puppet_arm_right_raw_action_chunk_deque[0].header.
                   stamp.to_sec() <= frame_time):
                op.puppet_arm_right_raw_action_chunk_deque.popleft()

        span_abs = abs(frame_time_max - frame_time)
        max_at = [
            n for n, ts in aligned_ts if abs(ts - frame_time_max) < _ts_eps
        ]
        min_at = [n for n, ts in aligned_ts if abs(ts - frame_time) < _ts_eps]

        fail_info = {
            'reason': 'span_exceed_slop',
            'wall_time': rospy.Time.now().to_sec(),
            'span': round(span_abs, 6),
            'sync_slop': op.sync_slop,
            'frame_time': round(frame_time, 6),
            'frame_time_max': round(frame_time_max, 6),
            'anchor_cams': anchor_cams_min,
            'max_at': max_at,
            'min_at': min_at,
            'cam_latest': {n: round(ts, 6)
                           for n, ts in img_latest_by_name},
            'aligned_ts': {n: round(ts, 6)
                           for n, ts in aligned_ts},
        }
        return (False, fail_info)

    # Align enabled image and puppet queues to frame_time before consuming.
    if cfg.img_head_topic:
        while (len(op.img_head_deque) > 0
               and op.img_head_deque[0].header.stamp.to_sec() < frame_time):
            op.img_head_deque.popleft()
    if cfg.img_left_topic:
        while (len(op.img_left_deque) > 0
               and op.img_left_deque[0].header.stamp.to_sec() < frame_time):
            op.img_left_deque.popleft()
    if cfg.img_right_topic:
        while (len(op.img_right_deque) > 0
               and op.img_right_deque[0].header.stamp.to_sec() < frame_time):
            op.img_right_deque.popleft()
    if use_front_camera:
        while (len(op.img_front_deque) > 0
               and op.img_front_deque[0].header.stamp.to_sec() < frame_time):
            op.img_front_deque.popleft()
    if cfg.puppet_arm_left_topic:
        while (len(op.puppet_arm_left_deque) > 0 and
               op.puppet_arm_left_deque[0].header.stamp.to_sec() < frame_time):
            op.puppet_arm_left_deque.popleft()
    if cfg.puppet_arm_right_topic:
        while (len(op.puppet_arm_right_deque) > 0
               and op.puppet_arm_right_deque[0].header.stamp.to_sec() <
               frame_time):
            op.puppet_arm_right_deque.popleft()
    if cfg.puppet_ee_pose_left_topic:
        while (len(op.puppet_ee_pose_left_deque) > 0
               and op.puppet_ee_pose_left_deque[0].header.stamp.to_sec() <
               frame_time):
            op.puppet_ee_pose_left_deque.popleft()
    if cfg.puppet_ee_pose_right_topic:
        while (len(op.puppet_ee_pose_right_deque) > 0
               and op.puppet_ee_pose_right_deque[0].header.stamp.to_sec() <
               frame_time):
            op.puppet_ee_pose_right_deque.popleft()

    # Raw action chunks are optional and consumed once. Use the last message
    # not newer than frame_time so chunks align with saved frame ids.
    puppet_arm_left_raw_action_chunk = None
    if (cfg.puppet_arm_left_raw_action_topic
            and len(op.puppet_arm_left_raw_action_chunk_deque) > 0):
        while (len(op.puppet_arm_left_raw_action_chunk_deque) > 0
               and op.puppet_arm_left_raw_action_chunk_deque[0].header.stamp.
               to_sec() <= frame_time):
            puppet_arm_left_raw_action_chunk = (
                op.puppet_arm_left_raw_action_chunk_deque.popleft())

    puppet_arm_right_raw_action_chunk = None
    if (cfg.puppet_arm_right_raw_action_topic
            and len(op.puppet_arm_right_raw_action_chunk_deque) > 0):
        while (len(op.puppet_arm_right_raw_action_chunk_deque) > 0
               and op.puppet_arm_right_raw_action_chunk_deque[0].header.stamp.
               to_sec() <= frame_time):
            puppet_arm_right_raw_action_chunk = (
                op.puppet_arm_right_raw_action_chunk_deque.popleft())

    # Convert images only after all synchronization checks pass.
    img_head = None
    img_head_ts = None
    if cfg.img_head_topic:
        head_msg = op.img_head_deque.popleft()
        img_head_ts = head_msg.header.stamp.to_sec()
        img_head = op.bridge.imgmsg_to_cv2(head_msg, 'bgr8')

    img_left = None
    img_left_ts = None
    if cfg.img_left_topic:
        left_msg = op.img_left_deque.popleft()
        img_left_ts = left_msg.header.stamp.to_sec()
        img_left = op.bridge.imgmsg_to_cv2(left_msg, 'bgr8')

    img_right = None
    img_right_ts = None
    if cfg.img_right_topic:
        right_msg = op.img_right_deque.popleft()
        img_right_ts = right_msg.header.stamp.to_sec()
        img_right = op.bridge.imgmsg_to_cv2(right_msg, 'bgr8')

    # Front camera is optional.
    img_front = None
    img_front_ts = None
    if use_front_camera:
        front_msg = op.img_front_deque.popleft()
        img_front_ts = front_msg.header.stamp.to_sec()
        img_front = op.bridge.imgmsg_to_cv2(front_msg, 'bgr8')

    # Puppet arm state is optional per configured topic.
    puppet_arm_left = None
    puppet_arm_left_ts = None
    if cfg.puppet_arm_left_topic:
        puppet_arm_left = op.puppet_arm_left_deque.popleft()
        puppet_arm_left_ts = puppet_arm_left.header.stamp.to_sec()

    puppet_arm_right = None
    puppet_arm_right_ts = None
    if cfg.puppet_arm_right_topic:
        puppet_arm_right = op.puppet_arm_right_deque.popleft()
        puppet_arm_right_ts = puppet_arm_right.header.stamp.to_sec()

    # End-pose streams are optional per configured topic.
    puppet_ee_pose_left = None
    puppet_ee_pose_left_ts = None
    if cfg.puppet_ee_pose_left_topic:
        puppet_ee_pose_left = op.puppet_ee_pose_left_deque.popleft()
        puppet_ee_pose_left_ts = puppet_ee_pose_left.header.stamp.to_sec()

    puppet_ee_pose_right = None
    puppet_ee_pose_right_ts = None
    if cfg.puppet_ee_pose_right_topic:
        puppet_ee_pose_right = op.puppet_ee_pose_right_deque.popleft()
        puppet_ee_pose_right_ts = puppet_ee_pose_right.header.stamp.to_sec()

    # Commands are not strict sync anchors and follow raw-action semantics.
    puppet_arm_left_cmd = None
    if (cfg.puppet_arm_left_cmd_topic
            and len(op.puppet_arm_left_cmd_deque) > 0):
        # Keep the latest command at or before frame_time.
        while (len(op.puppet_arm_left_cmd_deque) > 0
               and op.puppet_arm_left_cmd_deque[0].header.stamp.to_sec() <=
               frame_time):
            puppet_arm_left_cmd = op.puppet_arm_left_cmd_deque.popleft()

    puppet_arm_right_cmd = None
    if (cfg.puppet_arm_right_cmd_topic
            and len(op.puppet_arm_right_cmd_deque) > 0):
        while (len(op.puppet_arm_right_cmd_deque) > 0
               and op.puppet_arm_right_cmd_deque[0].header.stamp.to_sec() <=
               frame_time):
            puppet_arm_right_cmd = op.puppet_arm_right_cmd_deque.popleft()

    # Include timestamps for synchronized and opportunistically consumed data.
    timestamps: Dict[str, float] = {}
    # Synchronized streams.
    if cfg.img_head_topic:
        timestamps['img_head'] = img_head_ts
    if use_front_camera:
        timestamps['img_front'] = img_front_ts
    if cfg.img_left_topic:
        timestamps['img_left'] = img_left_ts
    if cfg.img_right_topic:
        timestamps['img_right'] = img_right_ts
    if cfg.puppet_arm_left_topic:
        timestamps['puppet_arm_left'] = puppet_arm_left_ts
    if cfg.puppet_arm_right_topic:
        timestamps['puppet_arm_right'] = puppet_arm_right_ts

    # End-pose timestamps.
    if cfg.puppet_ee_pose_left_topic and puppet_ee_pose_left is not None:
        timestamps['puppet_ee_pose_left'] = puppet_ee_pose_left_ts
    if cfg.puppet_ee_pose_right_topic and puppet_ee_pose_right is not None:
        timestamps['puppet_ee_pose_right'] = puppet_ee_pose_right_ts

    # Non-anchor streams.
    if puppet_arm_left_cmd is not None:
        timestamps['puppet_arm_left_cmd'] = (
            puppet_arm_left_cmd.header.stamp.to_sec())
    if puppet_arm_right_cmd is not None:
        timestamps['puppet_arm_right_cmd'] = (
            puppet_arm_right_cmd.header.stamp.to_sec())
    if puppet_arm_left_raw_action_chunk is not None:
        timestamps['puppet_arm_left_raw_action_chunk'] = (
            puppet_arm_left_raw_action_chunk.header.stamp.to_sec())
    if puppet_arm_right_raw_action_chunk is not None:
        timestamps['puppet_arm_right_raw_action_chunk'] = (
            puppet_arm_right_raw_action_chunk.header.stamp.to_sec())

    # Shared frame timestamp.
    timestamps['sync_time'] = frame_time

    result = FrameData(
        img_head=img_head,
        img_front=img_front,
        img_left=img_left,
        img_right=img_right,
        puppet_arm_left=puppet_arm_left,
        puppet_arm_right=puppet_arm_right,
        puppet_ee_pose_left=puppet_ee_pose_left,
        puppet_ee_pose_right=puppet_ee_pose_right,
        puppet_arm_left_cmd=puppet_arm_left_cmd,
        puppet_arm_right_cmd=puppet_arm_right_cmd,
        puppet_arm_left_raw_action_chunk=puppet_arm_left_raw_action_chunk,
        puppet_arm_right_raw_action_chunk=puppet_arm_right_raw_action_chunk,
        timestamps=timestamps,
    )
    return (True, result)
