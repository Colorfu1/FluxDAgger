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
"""Run offline Qwen3-VL reward inference on MP4/NPY data.

Usage:
    source <DAGGER_REPO>/devel/setup.bash
    python tools/reward_inference/qwen3_reward_inference.py \
        --src_dir <video_source_dir> \
        --save_mp4 <output_video.mp4> \
        --start <start_frame> --end <end_frame> \
        --model_path <qwen3_vl_reward_checkpoint>

Examples:
    python tools/reward_inference/qwen3_reward_inference.py \
        --src_dir /path/to/mp4_npy_dataset \
        --save_mp4 ./vis_rgb_labeled.mp4 \
        --start 0 --end 2880 \
        --f_interval 8 --t_interval 16 \
        --model_path /path/to/qwen3_vl_reward_checkpoint
"""

from __future__ import annotations
import argparse
import os
import time
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

from dagger.reward_models.qwen3_reward import (
    infer_once, parse_status, resolve_qwen3_attn_implementation)


def _read_frames_by_indices(video_path: str,
                            frame_indices: Sequence[int]) -> np.ndarray:
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f'Video not found: {video_path}')

    idx = np.asarray(frame_indices, dtype=np.int64)
    if idx.ndim != 1:
        raise ValueError('frame_indices must be a 1D list/array of integers.')
    if idx.size == 0:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f'Failed to open video: {video_path}')
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(f'Failed to read any frame from: {video_path}')
        h, w = frame.shape[:2]
        return np.empty((0, h, w, 3), dtype=np.uint8)

    if np.any(idx < 0):
        raise ValueError('frame_indices must be non-negative (0-based).')

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f'Failed to open video: {video_path}')

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count > 0 and np.any(idx >= frame_count):
        bad = idx[idx >= frame_count]
        cap.release()
        raise IndexError(
            f'Some requested frames are out of range for {video_path}. '
            f'max valid index={frame_count - 1}, got {bad.tolist()}')

    frames_rgb: List[np.ndarray] = []
    for i in idx.tolist():
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(i))
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            cap.release()
            raise RuntimeError(f'Failed to read frame {i} from {video_path}')
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames_rgb.append(frame_rgb)

    cap.release()
    return np.stack(frames_rgb, axis=0).astype(np.uint8, copy=False)


def read_rgb_triplet_frames(
    dir_path: str,
    frame_indices: Sequence[int],
    filenames: Tuple[str, str, str] = ('rgb.mp4', 'rgb_left_wrist.mp4',
                                       'rgb_right_wrist.mp4'),
) -> List[np.ndarray]:
    video_paths = [os.path.join(dir_path, fn) for fn in filenames]
    return [_read_frames_by_indices(vp, frame_indices) for vp in video_paths]


def get_video_meta(video_path: str) -> Tuple[int, int, float, int]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f'Failed to open video: {video_path}')
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if fps <= 0:
        fps = 24.0
    return w, h, fps, frame_count


def draw_label_on_rgb(frame_rgb: np.ndarray, status: str,
                      frame_idx: int) -> np.ndarray:
    img_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    label = f'{status} | frame={frame_idx}'
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2

    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    x, y = 10, 30
    cv2.rectangle(img_bgr, (x - 6, y - th - 6), (x + tw + 6, y + baseline + 6),
                  (0, 0, 0), -1)
    cv2.putText(img_bgr, label, (x, y), font, font_scale, (255, 255, 255),
                thickness, cv2.LINE_AA)

    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', type=str, required=True)
    parser.add_argument('--save_mp4', type=str, required=True)
    parser.add_argument('--start', type=int, required=True)
    parser.add_argument('--end', type=int, required=True)
    parser.add_argument('--f_interval', type=int, default=8)
    parser.add_argument('--t_interval', type=int, default=16)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument(
        '--prompt',
        type=str,
        default=('<video><video><video> Describe the task status: '
                 'fold the square off-white towel'),
    )
    parser.add_argument('--max_new_tokens', type=int, default=1024)
    args = parser.parse_args()

    rgb_path = os.path.join(args.src_dir, 'rgb.mp4')
    w, h, fps, frame_count = get_video_meta(rgb_path)

    if frame_count > 0:
        args.end = min(args.end, frame_count - 1)

    if args.start < 0 or args.start > args.end:
        raise ValueError(f'Invalid range: start={args.start}, end={args.end}')

    os.makedirs(os.path.dirname(args.save_mp4) or '.', exist_ok=True)

    load_kwargs = dict(
        attn_implementation=resolve_qwen3_attn_implementation(),
        device_map='auto',
    )
    try:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            **load_kwargs,
        )
    except TypeError:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model_path,
            dtype=torch.bfloat16,
            **load_kwargs,
        )
    processor = Qwen3VLProcessor.from_pretrained(args.model_path)

    total_frames = args.end - args.start + 1
    labels: List[
        Optional[str]] = [None] * total_frames  # index 0 -> frame=args.start

    t = args.start
    while t <= args.end:
        if t + 3 * args.f_interval > args.end:
            break

        frame_indices = [
            t,
            t + args.f_interval,
            t + 2 * args.f_interval,
            t + 3 * args.f_interval,
        ]

        video_triplet = read_rgb_triplet_frames(args.src_dir, frame_indices)
        start = time.time()
        output_text = infer_once(
            model=model,
            processor=processor,
            video_triplet=video_triplet,
            prompt_text=args.prompt,
            max_new_tokens=args.max_new_tokens,
        )
        print('inference time:', time.time() - start)
        status = parse_status(output_text)

        cover_l = t
        cover_r = min(t + args.t_interval - 1, args.end)

        for fi in range(cover_l, cover_r + 1):
            labels[fi - args.start] = status

        t += args.t_interval

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(args.save_mp4, fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f'Failed to open VideoWriter: {args.save_mp4}')

    cap = cv2.VideoCapture(rgb_path)
    if not cap.isOpened():
        writer.release()
        raise RuntimeError(f'Failed to open rgb video: {rgb_path}')

    cap.set(cv2.CAP_PROP_POS_FRAMES, float(args.start))

    cur = args.start
    while cur <= args.end:
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        status = labels[cur - args.start] if labels[
            cur - args.start] is not None else 'Normal'
        labeled_rgb = draw_label_on_rgb(
            frame_rgb, status=status, frame_idx=cur)

        if labeled_rgb.shape[0] != h or labeled_rgb.shape[1] != w:
            labeled_rgb = cv2.resize(
                labeled_rgb, (w, h), interpolation=cv2.INTER_AREA)

        writer.write(cv2.cvtColor(labeled_rgb, cv2.COLOR_RGB2BGR))
        cur += 1

    cap.release()
    writer.release()
    print(f'[OK] Saved visualization mp4 to: {args.save_mp4} '
          f'(fps={fps}, frames={cur - args.start})')


if __name__ == '__main__':
    main()
