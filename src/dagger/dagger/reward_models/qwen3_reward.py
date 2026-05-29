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
"""Core Qwen3-VL reward model logic without ROS dependencies."""

import os
from collections import deque
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np
import torch
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

from .base import BaseRewardModel

STATUS_REWARD_MAP = {
    'Normal': 1.0,
    'Static': 0.3,
    'Grasp fail': 0.0,
    'Shaking': 0.1,
    'Drop': 0.0,
}

DEFAULT_CAMERA_FIELDS = ('cam_high', 'cam_left_wrist', 'cam_right_wrist')


def parse_status(output_text: str) -> str:
    """Parse task status from model output text.

    Missing or unmatched output defaults to ``Normal``.
    """
    text = output_text or ''
    if 'Grasp fail' in text:
        return 'Grasp fail'
    if 'Static' in text:
        return 'Static'
    if 'Normal' in text:
        return 'Normal'
    if 'Reverse' in text:
        return 'Shaking'
    if 'drop' in text:
        return 'Drop'
    return 'Normal'


@torch.no_grad()
def infer_once(
    model: Qwen3VLForConditionalGeneration,
    processor: Qwen3VLProcessor,
    video_triplet: List[np.ndarray],
    prompt_text: str,
    max_new_tokens: int = 1024,
) -> str:
    """Run one Qwen3-VL inference pass over a three-view video clip.

    Args:
        model: Qwen3-VL model.
        processor: Qwen3-VL processor.
        video_triplet: Three RGB frame sequences shaped ``(N, H, W, 3)``.
        prompt_text: Text prompt.
        max_new_tokens: Maximum generated token count.

    Returns:
        Raw decoded model output.
    """
    messages = [{
        'role':
        'user',
        'content': [
            {
                'type': 'video',
                'video': video_triplet[0]
            },
            {
                'type': 'video',
                'video': video_triplet[1]
            },
            {
                'type': 'video',
                'video': video_triplet[2]
            },
            {
                'type': 'text',
                'text': prompt_text
            },
        ],
    }]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors='pt',
    ).to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    out = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return out[0] if out else ''


def resolve_qwen3_attn_implementation(explicit: Optional[str] = None) -> str:
    """Use FlashAttention2 when available, otherwise fall back to SDPA."""
    if explicit:
        return explicit
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        return 'sdpa'
    return 'flash_attention_2'


class Qwen3RewardModel(BaseRewardModel):
    """Qwen3-VL reward model for online frame-window inference."""

    _bridge: Any = None  # lazily initialised shared CvBridge instance

    def __init__(self) -> None:
        self.model = None
        self.processor = None

        self._frame_buffer_head: deque = deque()
        self._frame_buffer_left: deque = deque()
        self._frame_buffer_right: deque = deque()

        self._f_interval: int = 8
        self._t_interval: int = 16
        self._prompt: str = ''
        self._max_new_tokens: int = 1024
        self._source_tag: str = 'qwen3_reward'
        self._camera_fields: tuple[str, str, str] = DEFAULT_CAMERA_FIELDS

        self._last_infer_step: int = -999
        self._current_episode_id: int = -1

        self._status_history: List[str] = []

    def setup(self, config: Dict[str, Any]) -> None:
        """Load the Qwen3-VL model and processor.

        Expected config keys:
            - model_path (str): model checkpoint path.
            - prompt (str): inference prompt.
            - f_interval (int): frame interval.
            - t_interval (int): inference interval.
            - max_new_tokens (int): maximum generated token count.
            - source_tag (str): reward source identifier.
            - camera_fields (tuple/list[str]): SyncedObservation image fields.
            - attn_implementation (str, optional): explicit attention backend.
        """
        model_path = config['model_path']
        self._prompt = str(config['prompt'])
        self._f_interval = int(config.get('f_interval', 8))
        self._t_interval = int(config.get('t_interval', 16))
        self._max_new_tokens = int(config.get('max_new_tokens', 1024))
        self._source_tag = str(config.get('source_tag', 'qwen3_reward'))
        camera_fields = tuple(
            str(field)
            for field in config.get('camera_fields', DEFAULT_CAMERA_FIELDS))
        if len(camera_fields) != 3:
            raise ValueError('camera_fields must contain exactly 3 fields')
        self._camera_fields = (camera_fields[0], camera_fields[1],
                               camera_fields[2])

        _local_kw: Dict[str, Any] = {}
        if os.path.isdir(os.path.expanduser(model_path)):
            _local_kw['local_files_only'] = True

        _attn = resolve_qwen3_attn_implementation(
            config.get('attn_implementation'))

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            attn_implementation=_attn,
            device_map='auto',
            **_local_kw,
        )
        self.processor = Qwen3VLProcessor.from_pretrained(
            model_path,
            **_local_kw,
        )

    def on_episode_reset(self) -> None:
        """Clear frame buffers and status history at episode boundaries."""
        self._frame_buffer_head.clear()
        self._frame_buffer_left.clear()
        self._frame_buffer_right.clear()
        self._last_infer_step = -999
        self._status_history.clear()

    def predict(self, observation: Any) -> Optional[Dict[str, Any]]:
        """Process one observation and return reward output when ready.

        The observation is expected to match ``SyncedObservation`` fields:
            - episode_id, step_id
            - cam_high, cam_left_wrist, cam_right_wrist
              (``sensor_msgs/Image`` or numpy arrays)
            - is_terminal
        """
        ep_id = int(observation.episode_id)
        step_id = int(observation.step_id)

        if ep_id != self._current_episode_id:
            self.on_episode_reset()
            self._current_episode_id = ep_id

        if observation.is_terminal:
            self.on_episode_reset()
            return None

        img_head = self._extract_rgb(observation, self._camera_fields[0])
        img_left = self._extract_rgb(observation, self._camera_fields[1])
        img_right = self._extract_rgb(observation, self._camera_fields[2])

        if img_head is not None:
            self._frame_buffer_head.append((step_id, img_head))
        if img_left is not None:
            self._frame_buffer_left.append((step_id, img_left))
        if img_right is not None:
            self._frame_buffer_right.append((step_id, img_right))

        max_buf = self._f_interval * 4 + 10
        for buf in (self._frame_buffer_head, self._frame_buffer_left,
                    self._frame_buffer_right):
            while len(buf) > max_buf:
                buf.popleft()

        if step_id - self._last_infer_step < self._t_interval:
            return None

        target_steps = [
            step_id - 3 * self._f_interval,
            step_id - 2 * self._f_interval,
            step_id - 1 * self._f_interval,
            step_id,
        ]

        triplet = self._build_video_triplet(target_steps)
        if triplet is None:
            return None

        self._last_infer_step = step_id

        output_text = infer_once(
            model=self.model,
            processor=self.processor,
            video_triplet=triplet,
            prompt_text=self._prompt,
            max_new_tokens=self._max_new_tokens,
        )

        status = parse_status(output_text)
        self._status_history.append(status)
        reward_value = STATUS_REWARD_MAP.get(status, 0.5)

        return {
            'value': reward_value,
            'done': status in ('Grasp fail', 'Drop'),
            'confidence': reward_value,
            'aux_values': [],
            'source': self.source_tag,
            'status_text': status,
            'raw_output': output_text,
        }

    @property
    def source_tag(self) -> str:
        return self._source_tag

    @classmethod
    def _extract_rgb(cls, observation: Any,
                     camera_attr: str) -> Optional[np.ndarray]:
        """Extract one camera image from an observation as RGB.

        Supports ROS ``sensor_msgs/Image`` and numpy arrays.
        """
        img_msg = getattr(observation, camera_attr, None)
        if img_msg is None:
            return None

        if isinstance(img_msg, np.ndarray):
            if img_msg.ndim == 3 and img_msg.shape[2] == 3:
                return img_msg
            return None

        if not hasattr(img_msg, 'data') or not img_msg.data:
            return None

        try:
            from cv_bridge import CvBridge
            if cls._bridge is None:
                cls._bridge = CvBridge()
            bgr = cls._bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            return rgb
        except Exception:
            return None

    def _build_video_triplet(
            self, target_steps: Sequence[int]) -> Optional[List[np.ndarray]]:
        """Build four-frame video clips for the three camera views.

        Returns:
            ``[head_frames, left_frames, right_frames]`` as uint8 arrays, or
            ``None`` when the buffers are incomplete.
        """
        result = []
        for buf in (self._frame_buffer_head, self._frame_buffer_left,
                    self._frame_buffer_right):
            if not buf:
                return None
            frames = []
            buf_steps = [s for s, _ in buf]
            for ts in target_steps:
                best_idx = min(
                    range(len(buf_steps)),
                    key=lambda i: abs(buf_steps[i] - ts))
                frames.append(buf[best_idx][1])
            result.append(
                np.stack(frames, axis=0).astype(np.uint8, copy=False))
        return result
