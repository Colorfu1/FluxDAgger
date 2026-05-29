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
"""DAgger package for multi-arm data collection and online evaluation.

Package layout::

    dagger/
        runtime_config.py           YAML-backed runtime configuration loader
        infra/                      ROS I/O infrastructure
            ros_observation_buffer.py
                                      ROS observation subscriptions + queues
            frame_sync.py           sync_get_frame algorithm + FrameData
            async_worker.py         InferenceWorker
        hardware/
            piper_arm.py            Piper arm SDK wrapper
        collectors/
            sync_frame_collector.py parquet episode writer
        reward_models/
            base.py                 BaseRewardModel interface
            qwen3_reward.py         Qwen3-VL implementation

Node entry points live in ``src/dagger/scripts/`` and are installed through
``catkin_install_python``.
"""

from __future__ import annotations

__all__: list[str] = []
