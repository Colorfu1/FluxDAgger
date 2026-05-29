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
"""Convert FluxDAgger parquet episodes into MP4 videos and NumPy arrays.

Usage:
    source <DAGGER_REPO>/devel/setup.bash
    python tools/data_processing/parquet_to_mp4_npy.py \
        --input_base <raw_dataset_dir> \
        --output_base <mp4_npy_dataset_dir> [--fps 30]

Examples:
    python tools/data_processing/parquet_to_mp4_npy.py \
        --input_base /path/to/raw_dataset \
        --output_base /path/to/mp4_npy_dataset \
        --fps 30
"""

from __future__ import annotations
import argparse
import gc
import itertools
import json
from pathlib import Path
from typing import Dict, Iterable, List

import av
import cv2
import numpy as np
import pandas as pd
from episode_utils import (category_key_from_path, crop_sensor_index,
                           done_flag_path_for, get_episode_items,
                           load_sensor_index, normalize_episode_name,
                           split_frames_by_mode)
from tqdm import tqdm

from dagger.runtime_config import load_runtime_config

_CONFIG = load_runtime_config()
CAM_FRONT = _CONFIG.camera_name('front')
CAM_HIGH = _CONFIG.camera_name('head')
CAM_LEFT_WRIST = _CONFIG.camera_name('left_wrist')
CAM_RIGHT_WRIST = _CONFIG.camera_name('right_wrist')


def save_frames_and_encode_video(frames: Iterable[np.ndarray],
                                 video_path: str,
                                 fps: int,
                                 vcodec: str = 'libsvtav1',
                                 fix_color_order: bool = False):
    """Encode video directly from a frame iterable.

    Falls back to h264 when the requested encoder is unavailable.

    Args:
        frames: Iterable of decoded image frames.
        video_path: Output video path.
        fps: Output frame rate.
        vcodec: Preferred video codec.
        fix_color_order: Convert BGR/BGRA frames to RGB before encoding.
    """
    frame_iter = iter(frames)
    try:
        first_frame = next(frame_iter)
    except StopIteration:
        raise ValueError('frames is empty; cannot encode video')

    if first_frame is None:
        raise ValueError('first frame is None; cannot encode video')

    height, width = first_frame.shape[:2]

    codecs_to_try = [vcodec]
    if vcodec != 'h264':
        codecs_to_try.append('h264')

    last_error = None
    for current_codec in codecs_to_try:
        try:
            with av.open(video_path, 'w') as output:
                if current_codec == 'libsvtav1':
                    options = {'g': '2', 'crf': '30'}
                elif current_codec == 'h264':
                    options = {'g': '2', 'crf': '23'}
                else:
                    options = {'g': '2', 'crf': '30'}

                stream = output.add_stream(current_codec, fps, options=options)
                stream.pix_fmt = 'yuv420p'
                stream.width = width
                stream.height = height

                for frame in itertools.chain([first_frame], frame_iter):
                    if len(frame.shape) == 3 and frame.shape[2] == 3:
                        if frame.dtype != np.uint8:
                            frame = frame.astype(np.uint8)

                        if fix_color_order:
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        else:
                            frame_rgb = frame
                    elif len(frame.shape) == 3 and frame.shape[2] == 4:
                        if fix_color_order:
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
                        else:
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
                    else:
                        frame_rgb = frame

                    if frame_rgb.dtype != np.uint8:
                        frame_rgb = frame_rgb.astype(np.uint8)

                    video_frame = av.VideoFrame.from_ndarray(
                        frame_rgb, format='rgb24')
                    for packet in stream.encode(video_frame):
                        output.mux(packet)

                for packet in stream.encode():
                    output.mux(packet)

            if current_codec != vcodec:
                print(f'  Note: {vcodec} is unavailable; using '
                      f'{current_codec} instead')
            return

        except (av.codec.codec.UnknownCodecError, ValueError) as e:
            last_error = e
            continue
        except Exception as e:
            raise RuntimeError(
                f'Video encoding failed '
                f'(codec={current_codec}, path={video_path}): {e}') from e

    raise RuntimeError(
        f'Video encoding failed: no codec is available '
        f'({", ".join(codecs_to_try)})\n'
        f'Last error: {last_error}\n'
        f'Hint: install ffmpeg and related codecs, or check the environment.')


def export_segment_to_dir(df: pd.DataFrame,
                          output_dir: Path,
                          sensor_data: Dict,
                          selected_indices: List[int],
                          success_override: bool = None,
                          score_override: float = None,
                          fps: int = 30,
                          vcodec: str = 'libsvtav1') -> None:
    """Export one DataFrame segment to mp4, npy, and ``info.json``."""
    output_dir.mkdir(parents=True, exist_ok=True)

    qpos = None
    if '/observations/qpos' in df.columns:
        qpos = np.array(df['/observations/qpos'].tolist())
        np.save(str(output_dir / 'qpos.npy'), qpos)

    eepose = None
    if '/observations/eepose' in df.columns:
        eepose = np.array(df['/observations/eepose'].tolist())
        np.save(str(output_dir / 'eepose.npy'), eepose)

    actions = None
    if '/action' in df.columns:
        actions = np.array(df['/action'].tolist())
        np.save(str(output_dir / 'action.npy'), actions)

    if '/observations/qvel' in df.columns:
        qvel = np.array(df['/observations/qvel'].tolist())
        np.save(str(output_dir / 'qvel.npy'), qvel)

    if '/observations/effort' in df.columns:
        effort = np.array(df['/observations/effort'].tolist())
        np.save(str(output_dir / 'effort.npy'), effort)

    if '/base_action' in df.columns:
        base_action = np.array(df['/base_action'].tolist())
        np.save(str(output_dir / 'base_action.npy'), base_action)

    camera_names: List[str] = []
    width = 0
    height = 0

    for col in df.columns:
        if not col.startswith('/observations/images/'):
            continue

        cam_name = col.split('/')[-1]
        frames_bytes = df[col].tolist()

        first_valid_img = None
        total_valid_frames = 0
        for fb in frames_bytes:
            if fb is None:
                continue
            buf = np.frombuffer(fb, dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img is None:
                continue
            total_valid_frames += 1
            if first_valid_img is None:
                first_valid_img = img

        if first_valid_img is None:
            print(f'  Warning: {cam_name} has no decodable frames; skip it')
            continue

        if cam_name == CAM_HIGH or cam_name == 'cam_high_high':
            video_path = output_dir / 'rgb.mp4'
            camera_names.append('rgb')
        elif cam_name == CAM_FRONT:
            video_path = output_dir / 'rgb_front.mp4'
            camera_names.append('rgb_front')
        elif cam_name == CAM_LEFT_WRIST:
            video_path = output_dir / 'rgb_left_wrist.mp4'
            camera_names.append('rgb_left_wrist')
        elif cam_name == CAM_RIGHT_WRIST:
            video_path = output_dir / 'rgb_right_wrist.mp4'
            camera_names.append('rgb_right_wrist')
        else:
            print(f'  Unknown camera: {cam_name}, skipped')
            continue

        height, width, _ = first_valid_img.shape
        print(f'  {cam_name}: {total_valid_frames} frames, {width}x{height}')

        def frame_generator(bytes_list):
            for fb in bytes_list:
                if fb is None:
                    continue
                buf_inner = np.frombuffer(fb, dtype=np.uint8)
                img_inner = cv2.imdecode(buf_inner, cv2.IMREAD_COLOR)
                if img_inner is not None:
                    yield img_inner

        # Parquet PNGs are encoded from OpenCV BGR arrays (see
        # sync_frame_collector); imdecode returns BGR. PyAV expects rgb24.
        save_frames_and_encode_video(
            frame_generator(frames_bytes),
            str(video_path),
            fps,
            fix_color_order=True,
            vcodec=vcodec,
        )

        del frames_bytes
        gc.collect()

    cropped_sensor = crop_sensor_index(
        sensor_data,
        selected_indices=selected_indices,
        segment_len=len(df),
        success_override=success_override,
        score_override=score_override,
    )

    episode_name = output_dir.name
    json_data = {
        'id': episode_name,
        'episode_id': 0,
        'trajectory_type': sensor_data.get('trajectory_type', 'rollout'),
        'success': cropped_sensor.get('success', False),
        'score': float(cropped_sensor.get('score', 0.0)),
        'timestamp': sensor_data.get('timestamp', ''),
        'collection_mode': sensor_data.get('collection_mode', 'synced'),
        'save_format': 'parquet',
        'frame_count': cropped_sensor.get('frame_count', len(df)),
        'frames': cropped_sensor.get('frame_count', len(df)),
        'fps': fps,
        'width': width,
        'height': height,
        'cameras': camera_names,
        'frame_mode_mapping': cropped_sensor.get('frame_mode_mapping', {}),
        'sensor_index': cropped_sensor.get('sensor_index', {}),
        'reward': 1.0 if cropped_sensor.get('success', False) else 0.0,
    }

    with (output_dir / 'info.json').open('w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)


def process_episode_to_segments(parquet_path: Path,
                                sensor_index_path: Path,
                                output_base_dir: Path,
                                fps: int = 30,
                                category_key: str = 'root',
                                vcodec: str = 'libsvtav1') -> None:
    """Split one episode into human/rollout segments and export mp4/npy."""
    sensor_data = load_sensor_index(sensor_index_path)
    df = pd.read_parquet(parquet_path, engine='pyarrow')

    frame_count = len(df)
    print(f'  Read {parquet_path.name}, frames: {frame_count}')

    segments, has_human = split_frames_by_mode(sensor_data, frame_count)
    success = sensor_data.get('success', False)
    episode_score = sensor_data.get('score', 0.0)

    base_episode_name = normalize_episode_name(parquet_path)

    human_indices: List[int] = []
    rollout_indices: List[int] = []
    for seg in segments:
        if seg['kind'] == 'human':
            human_indices.extend(seg['indices'])
        else:
            rollout_indices.extend(seg['indices'])

    if human_indices:
        human_df = df.iloc[human_indices].reset_index(drop=True)
        human_out_dir = output_base_dir / 'human' / base_episode_name
        print(f'  Export merged human segment: {len(human_df)} frames -> '
              f'{human_out_dir}')
        export_segment_to_dir(
            human_df,
            human_out_dir,
            sensor_data,
            selected_indices=human_indices,
            success_override=None,
            score_override=episode_score,
            fps=fps,
            vcodec=vcodec,
        )

    if rollout_indices:
        rollout_df = df.iloc[rollout_indices].reset_index(drop=True)

        if has_human:
            rollout_out_dir = (
                output_base_dir / 'rollout' / 'fail' / base_episode_name)
            success_override = False
            print(f'  Export rollout/fail segment from non-human frames: '
                  f'{len(rollout_df)} frames -> {rollout_out_dir}')
        else:
            if success:
                rollout_out_dir = (
                    output_base_dir / 'rollout' / 'success' /
                    base_episode_name)
                success_override = True
                print(f'  No human intervention, successful rollout: '
                      f'{len(rollout_df)} frames -> {rollout_out_dir}')
            else:
                rollout_out_dir = (
                    output_base_dir / 'rollout' / 'fail' / base_episode_name)
                success_override = False
                print(f'  No human intervention, failed rollout: '
                      f'{len(rollout_df)} frames -> {rollout_out_dir}')

        export_segment_to_dir(
            rollout_df,
            rollout_out_dir,
            sensor_data,
            selected_indices=rollout_indices,
            success_override=success_override,
            score_override=(0.0 if has_human else None),
            fps=fps,
            vcodec=vcodec,
        )

    done_flag_path = done_flag_path_for(output_base_dir, category_key,
                                        base_episode_name)
    done_flag_path.parent.mkdir(parents=True, exist_ok=True)
    with done_flag_path.open('w', encoding='utf-8') as f:
        f.write('OK\n')

    del df
    gc.collect()


def process_dagger_episodes_to_mp4(input_base_dir: Path,
                                   output_base_dir: Path,
                                   fps: int = 30,
                                   vcodec: str = 'libsvtav1') -> None:
    """Batch-export synced parquet episodes to human/rollout mp4/npy data."""
    episode_items = get_episode_items(input_base_dir)
    print(f'\nFound {len(episode_items)} episodes under {input_base_dir}')

    if not episode_items:
        return

    for episode_dir_name, parquet_path, sensor_index_path in tqdm(
            episode_items, desc='Processing episodes', unit='episode'):
        base_episode_name = normalize_episode_name(parquet_path)
        category_key = category_key_from_path(input_base_dir, parquet_path)

        done_flag_path = done_flag_path_for(output_base_dir, category_key,
                                            base_episode_name)
        if done_flag_path.exists():
            print(f'\nSkip already processed {episode_dir_name} '
                  f'(category={category_key}, '
                  f'episode={base_episode_name})')
            continue

        print(f'\nProcessing {episode_dir_name}')
        process_episode_to_segments(
            parquet_path,
            sensor_index_path,
            output_base_dir,
            fps,
            category_key=category_key,
            vcodec=vcodec,
        )


def main():
    parser = argparse.ArgumentParser(
        description=('Split synced parquet episodes into human, '
                     'rollout-success, and rollout-fail mp4/npy data.'))
    parser.add_argument(
        '--input_base',
        type=str,
        required=True,
        help='Input base directory containing episode_XXX directories or '
        'episode_XXX.parquet files.',
    )
    parser.add_argument(
        '--output_base',
        type=str,
        required=True,
        help='Output base directory; human and rollout subdirectories are '
        'created inside it.',
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=30,
        help='Output video FPS. Defaults to 30.',
    )
    parser.add_argument(
        '--vcodec',
        type=str,
        default='libsvtav1',
        help=('PyAV encoder name (e.g. libsvtav1, h264). Use h264 when '
              'downstream tools read MP4 via OpenCV VideoCapture.'),
    )

    args = parser.parse_args()

    input_base = Path(args.input_base)
    output_base = Path(args.output_base)

    if not input_base.exists():
        print(f'Error: input directory does not exist: {input_base}')
        return

    output_base.mkdir(parents=True, exist_ok=True)

    print(f'Input directory: {input_base}')
    print(f'Output directory: {output_base}')

    process_dagger_episodes_to_mp4(
        input_base,
        output_base,
        fps=args.fps,
        vcodec=args.vcodec,
    )
    print('All episodes processed.')


if __name__ == '__main__':
    main()
