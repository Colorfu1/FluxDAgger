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
"""ROS subscribers and bounded queues for synchronized observations.

``RosObservationBuffer.get_frame`` delegates timestamp alignment to
``dagger.infra.frame_sync.sync_get_frame``. This module owns ROS subscriptions
and message queues; the sync algorithm lives in ``frame_sync``.
"""

from __future__ import annotations
import threading
from collections import deque
from dataclasses import dataclass

import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image, JointState
from trajectory_msgs.msg import JointTrajectory

from dagger.infra.dagger_log import get_logger
from dagger.infra.frame_sync import FrameData, sync_get_frame
from dagger.runtime_config import (RuntimeConfig, get_ros_override,
                                   get_ros_param)

logger = get_logger(__name__, prefix='RosObservationBuffer')

__all__ = ['FrameData', 'RosObservationBuffer', 'RosObservationBufferConfig']


def _subscribe_if_enabled(topic: str, msg_type, callback) -> None:
    """Create a subscriber only when ``topic`` is non-empty."""
    if not topic:
        return
    rospy.Subscriber(
        topic,
        msg_type,
        callback,
        queue_size=1000,
        tcp_nodelay=True,
    )


@dataclass(frozen=True)
class RosObservationBufferConfig:
    """Topics and sync options for :class:`RosObservationBuffer`.

    Empty topic strings disable the matching subscription.
    """

    # RGB cameras.
    img_head_topic: str = ''
    img_front_topic: str = ''
    img_left_topic: str = ''
    img_right_topic: str = ''

    # Puppet arm state and policy command streams.
    puppet_arm_left_topic: str = ''
    puppet_arm_right_topic: str = ''
    puppet_arm_left_cmd_topic: str = ''
    puppet_arm_right_cmd_topic: str = ''
    puppet_arm_left_raw_action_topic: str = ''
    puppet_arm_right_raw_action_topic: str = ''
    puppet_ee_pose_left_topic: str = ''
    puppet_ee_pose_right_topic: str = ''

    debug: bool = False
    sync_slop: float = 0.05

    @classmethod
    def from_runtime_config(
        cls,
        runtime_config: RuntimeConfig,
    ) -> 'RosObservationBufferConfig':
        """Build sync defaults from the unified runtime config."""
        camera_topics = runtime_config.camera_sync_topics()
        topics = runtime_config.topics
        return cls(
            img_head_topic=camera_topics.get('head', ''),
            img_front_topic=camera_topics.get('front', ''),
            img_left_topic=camera_topics.get('left_wrist', ''),
            img_right_topic=camera_topics.get('right_wrist', ''),
            puppet_arm_left_topic=topics.puppet_arm_left,
            puppet_arm_right_topic=topics.puppet_arm_right,
            puppet_arm_left_cmd_topic=topics.master_joint_left,
            puppet_arm_right_cmd_topic=topics.master_joint_right,
            puppet_arm_left_raw_action_topic=topics.raw_action_left,
            puppet_arm_right_raw_action_topic=topics.raw_action_right,
            puppet_ee_pose_left_topic=topics.puppet_ee_pose_left,
            puppet_ee_pose_right_topic=topics.puppet_ee_pose_right,
            debug=runtime_config.runtime.debug,
            sync_slop=runtime_config.runtime.sync_slop,
        )

    @classmethod
    def from_ros_params(
        cls,
        ns: str = '~',
        runtime_config: RuntimeConfig | None = None,
    ) -> 'RosObservationBufferConfig':
        """Build config from ROS params matching the dataclass field names."""
        if runtime_config is None:
            raise ValueError('runtime_config is required')
        defaults = cls.from_runtime_config(runtime_config)
        return cls(
            img_head_topic=get_ros_param(
                ns=ns, name='img_head_topic', default=defaults.img_head_topic),
            img_front_topic=get_ros_param(
                ns=ns,
                name='img_front_topic',
                default=defaults.img_front_topic),
            img_left_topic=get_ros_param(
                ns=ns, name='img_left_topic', default=defaults.img_left_topic),
            img_right_topic=get_ros_param(
                ns=ns,
                name='img_right_topic',
                default=defaults.img_right_topic),
            puppet_arm_left_topic=get_ros_param(
                ns=ns,
                name='puppet_arm_left_topic',
                default=defaults.puppet_arm_left_topic),
            puppet_arm_right_topic=get_ros_param(
                ns=ns,
                name='puppet_arm_right_topic',
                default=defaults.puppet_arm_right_topic),
            puppet_arm_left_cmd_topic=get_ros_param(
                ns=ns,
                name='puppet_arm_left_cmd_topic',
                default=defaults.puppet_arm_left_cmd_topic),
            puppet_arm_right_cmd_topic=get_ros_param(
                ns=ns,
                name='puppet_arm_right_cmd_topic',
                default=defaults.puppet_arm_right_cmd_topic),
            puppet_arm_left_raw_action_topic=get_ros_param(
                ns=ns,
                name='puppet_arm_left_raw_action_topic',
                default=defaults.puppet_arm_left_raw_action_topic),
            puppet_arm_right_raw_action_topic=get_ros_param(
                ns=ns,
                name='puppet_arm_right_raw_action_topic',
                default=defaults.puppet_arm_right_raw_action_topic),
            puppet_ee_pose_left_topic=get_ros_param(
                ns=ns,
                name='puppet_ee_pose_left_topic',
                default=defaults.puppet_ee_pose_left_topic),
            puppet_ee_pose_right_topic=get_ros_param(
                ns=ns,
                name='puppet_ee_pose_right_topic',
                default=defaults.puppet_ee_pose_right_topic),
            debug=get_ros_override(
                ns=ns, name='debug', default=defaults.debug, value_type=bool),
            sync_slop=get_ros_override(
                ns=ns,
                name='sync_slop',
                default=defaults.sync_slop,
                value_type=float),
        )


class RosObservationBuffer:
    """Subscribe to observation topics and keep bounded message queues."""

    def __init__(self, config: RosObservationBufferConfig) -> None:
        self.config = config
        self.puppet_arm_right_deque = None
        self.puppet_arm_left_deque = None
        self.img_front_deque = None
        self.img_right_deque = None
        self.img_left_deque = None
        self.bridge = None
        # Shared max length for all sensor queues.
        self.max_queue_size = 1000
        # Last queue-full warning time, keyed by queue name.
        self.last_warn_time: dict = {}
        self.warn_interval = 5.0
        # Collection toggles this flag to accept/drop incoming messages.
        self.accept_new_data = False
        self.accept_new_data_lock = threading.Lock()
        # Cached scalar config values used in hot paths.
        self.is_debug_enabled = bool(config.debug)
        self.sync_slop = float(config.sync_slop)
        self.init()
        self.init_ros()

    def init(self):
        self.bridge = CvBridge()
        # Bounded queue objects drop the oldest message on append.
        make_deque = lambda: deque(maxlen=self.max_queue_size)  # noqa: E731
        self.img_left_deque = make_deque()
        self.img_right_deque = make_deque()
        self.img_front_deque = make_deque()
        self.img_head_deque = make_deque()
        self.puppet_arm_left_deque = make_deque()
        self.puppet_arm_right_deque = make_deque()
        self.puppet_arm_left_cmd_deque = make_deque()
        self.puppet_arm_right_cmd_deque = make_deque()
        self.puppet_arm_left_raw_action_chunk_deque = make_deque()
        self.puppet_arm_right_raw_action_chunk_deque = make_deque()
        self.puppet_ee_pose_left_deque = make_deque()
        self.puppet_ee_pose_right_deque = make_deque()

    def _should_log_warn(self, queue_name):
        """Return whether a queue-full warning should be emitted."""
        current_time = rospy.Time.now().to_sec()
        if queue_name not in self.last_warn_time:
            self.last_warn_time[queue_name] = current_time
            return True
        if (current_time - self.last_warn_time[queue_name] >=
                self.warn_interval):
            self.last_warn_time[queue_name] = current_time
            return True
        return False

    def get_frame(self) -> 'tuple[bool, FrameData | str | dict]':
        """Return one synchronized observation frame."""
        return sync_get_frame(self)

    def _enqueue_msg(self, deque_obj, msg, label, item_suffix='frame'):
        """Shared ROS subscriber callback implementation.

        All queues are bounded deque objects, so append handles eviction.
        """
        with self.accept_new_data_lock:
            if not self.accept_new_data:
                return
        if (self.is_debug_enabled and len(deque_obj) >= self.max_queue_size
                and self._should_log_warn(label)):
            logger.warning(
                '%s is full (size=%d), dropping oldest %s',
                label,
                len(deque_obj),
                item_suffix,
            )
        deque_obj.append(msg)

    # Callback proxies keep stable method names for frame_sync.

    def img_left_callback(self, msg):
        self._enqueue_msg(self.img_left_deque, msg, 'img_left_deque')

    def img_right_callback(self, msg):
        self._enqueue_msg(self.img_right_deque, msg, 'img_right_deque')

    def img_front_callback(self, msg):
        self._enqueue_msg(self.img_front_deque, msg, 'img_front_deque')

    def img_head_callback(self, msg):
        self._enqueue_msg(self.img_head_deque, msg, 'img_head_deque')

    def puppet_arm_left_callback(self, msg):
        self._enqueue_msg(self.puppet_arm_left_deque, msg,
                          'puppet_arm_left_deque', 'state')

    def puppet_arm_right_callback(self, msg):
        self._enqueue_msg(self.puppet_arm_right_deque, msg,
                          'puppet_arm_right_deque', 'state')

    def puppet_arm_left_cmd_callback(self, msg):
        self._enqueue_msg(self.puppet_arm_left_cmd_deque, msg,
                          'puppet_arm_left_cmd_deque', 'cmd')

    def puppet_arm_right_cmd_callback(self, msg):
        self._enqueue_msg(self.puppet_arm_right_cmd_deque, msg,
                          'puppet_arm_right_cmd_deque', 'cmd')

    def puppet_arm_left_raw_action_chunk_callback(self, msg):
        self._enqueue_msg(
            self.puppet_arm_left_raw_action_chunk_deque,
            msg,
            'puppet_arm_left_raw_action_chunk_deque',
            'raw action chunk',
        )

    def puppet_arm_right_raw_action_chunk_callback(self, msg):
        self._enqueue_msg(
            self.puppet_arm_right_raw_action_chunk_deque,
            msg,
            'puppet_arm_right_raw_action_chunk_deque',
            'raw action chunk',
        )

    def puppet_ee_pose_left_callback(self, msg):
        """Handle left puppet end-pose messages."""
        self._enqueue_msg(self.puppet_ee_pose_left_deque, msg,
                          'puppet_ee_pose_left_deque', 'pose')

    def puppet_ee_pose_right_callback(self, msg):
        """Handle right puppet end-pose messages."""
        self._enqueue_msg(self.puppet_ee_pose_right_deque, msg,
                          'puppet_ee_pose_right_deque', 'pose')

    def init_ros(self):
        """Subscribe to every topic enabled in the buffer config."""
        cfg = self.config
        sub = _subscribe_if_enabled

        # RGB cameras.
        sub(cfg.img_head_topic, Image, self.img_head_callback)
        sub(cfg.img_left_topic, Image, self.img_left_callback)
        sub(cfg.img_right_topic, Image, self.img_right_callback)
        sub(cfg.img_front_topic, Image, self.img_front_callback)

        # Puppet arm state, policy commands, and raw action chunks.
        sub(cfg.puppet_arm_left_topic, JointState,
            self.puppet_arm_left_callback)
        sub(cfg.puppet_arm_right_topic, JointState,
            self.puppet_arm_right_callback)
        sub(cfg.puppet_arm_left_cmd_topic, JointState,
            self.puppet_arm_left_cmd_callback)
        sub(cfg.puppet_arm_right_cmd_topic, JointState,
            self.puppet_arm_right_cmd_callback)
        sub(cfg.puppet_arm_left_raw_action_topic, JointTrajectory,
            self.puppet_arm_left_raw_action_chunk_callback)
        sub(cfg.puppet_arm_right_raw_action_topic, JointTrajectory,
            self.puppet_arm_right_raw_action_chunk_callback)

        # End poses.
        sub(cfg.puppet_ee_pose_left_topic, PoseStamped,
            self.puppet_ee_pose_left_callback)
        sub(cfg.puppet_ee_pose_right_topic, PoseStamped,
            self.puppet_ee_pose_right_callback)
