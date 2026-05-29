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
"""Replay a saved qpos.npy sequence to ROS joint command topics.

Usage:
    source <DAGGER_REPO>/devel/setup.bash
    python tools/replay_qpos_npy.py \
        --qpos_path <qpos.npy> [--frame_rate 30]

Examples:
    python tools/replay_qpos_npy.py \
        --qpos_path /path/to/qpos.npy \
        --left_topic /master/joint_left \
        --right_topic /master/joint_right \
        --frame_rate 30
"""

import argparse
import os

import numpy as np
import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Header

from dagger.runtime_config import load_runtime_config

_CONFIG = load_runtime_config()


def load_qpos_npy(qpos_path):
    """Load qpos.npy and validate the expected arm-state layout."""
    if not os.path.exists(qpos_path):
        print(f'Error: qpos file does not exist: {qpos_path}')
        exit(1)

    qpos = np.load(qpos_path)

    # Expected shape: (num_frames, 14), left arm first then right arm.
    if len(qpos.shape) != 2 or qpos.shape[1] != 14:
        print(f'Error: invalid qpos shape, expected (num_frames, 14), '
              f'got {qpos.shape}')
        exit(1)

    print(f'Loaded qpos data: {len(qpos)} frames')
    return qpos


def main(args):
    rospy.init_node('simple_qpos_publisher', anonymous=True)

    left_pub = rospy.Publisher(args.left_topic, JointState, queue_size=10)
    right_pub = rospy.Publisher(args.right_topic, JointState, queue_size=10)

    # Reuse one JointState message and update only the timestamp/position.
    joint_msg = JointState()
    joint_msg.header = Header()
    joint_msg.name = [
        'joint0',
        'joint1',
        'joint2',
        'joint3',
        'joint4',
        'joint5',
        'joint6',
    ]
    joint_msg.velocity = []
    joint_msg.effort = []

    rate = rospy.Rate(args.frame_rate)

    qpos_sequence = load_qpos_npy(args.qpos_path)

    # Replay frames directly, without interpolation.
    print(f'Start publishing qpos data at {args.frame_rate} Hz...')
    for frame_idx, qpos in enumerate(qpos_sequence):
        if rospy.is_shutdown():
            print('ROS node is shutting down; stop publishing')
            break

        joint_msg.header.stamp = rospy.Time.now()
        joint_msg.header.seq = frame_idx

        joint_msg.position = qpos[:7].tolist()
        left_pub.publish(joint_msg)

        joint_msg.position = qpos[7:].tolist()
        right_pub.publish(joint_msg)

        # Log every few frames to keep the terminal readable.
        if frame_idx % 10 == 0:
            left_str = np.round(qpos[:7], 4)
            right_str = np.round(qpos[7:], 4)
            print(f'Published frame {frame_idx:04d} | '
                  f'left: {left_str} | right: {right_str}')

        rate.sleep()

    print(f'Publish complete: {len(qpos_sequence)} frames')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Simple qpos publisher without interpolation.')

    parser.add_argument(
        '--qpos_path', type=str, required=True, help='Path to qpos.npy.')

    parser.add_argument(
        '--left_topic',
        type=str,
        default=_CONFIG.topics.puppet_arm_left,
        help='Left-arm publish topic.')
    parser.add_argument(
        '--right_topic',
        type=str,
        default=_CONFIG.topics.puppet_arm_right,
        help='Right-arm publish topic.')

    parser.add_argument(
        '--frame_rate', type=int, default=30, help='Publish rate in Hz.')

    args = parser.parse_args()
    try:
        main(args)
    except rospy.ROSInterruptException:
        print('Program interrupted by ROS.')
    except Exception as e:
        print(f'Program error: {e}')
