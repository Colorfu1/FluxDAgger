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
"""Split synced parquet episodes into human/rollout parquet subsets.

Usage:
    python tools/data_processing/split_parquet_by_mode.py \
        --input_base <raw_dataset_dir> \
        --output_base <sub_parquet_dataset_dir>

Examples:
    python tools/data_processing/split_parquet_by_mode.py \
        --input_base /path/to/raw_dataset \
        --output_base /path/to/sub_parquet_dataset
"""

from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from episode_utils import (category_key_from_path, crop_sensor_index,
                           done_flag_path_for, get_episode_items,
                           load_sensor_index, normalize_episode_name,
                           split_frames_by_mode)
from tqdm import tqdm


def export_segment_parquet(
    df: pd.DataFrame,
    episode_dir: Path,
    episode_number: str,
    sensor_data: Dict,
    selected_indices: List[int],
    success_override: Optional[bool] = None,
    score_override: Optional[float] = None,
) -> None:
    """Export one segment to parquet with a cropped ``sensor_index.json``."""
    episode_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = episode_dir / f'episode_{episode_number}.parquet'
    df.to_parquet(parquet_path, engine='pyarrow', index=False)

    cropped = crop_sensor_index(
        sensor_data,
        selected_indices=selected_indices,
        segment_len=len(df),
        success_override=success_override,
        score_override=score_override,
    )
    sensor_index_path = episode_dir / 'sensor_index.json'
    with sensor_index_path.open('w', encoding='utf-8') as f:
        json.dump(cropped, f, indent=2, ensure_ascii=False)


def process_episode_to_parquet_segments(
    parquet_path: Path,
    sensor_index_path: Path,
    output_base_dir: Path,
    category_key: str = 'root',
) -> None:
    """Split one episode into human/rollout parquet segments."""
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
        print(f'  Export human segment: {len(human_df)} frames -> '
              f'{human_out_dir}')
        export_segment_parquet(
            human_df,
            human_out_dir,
            base_episode_name.replace('episode_', ''),
            sensor_data,
            selected_indices=human_indices,
            success_override=None,
            score_override=episode_score,
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

        export_segment_parquet(
            rollout_df,
            rollout_out_dir,
            base_episode_name.replace('episode_', ''),
            sensor_data,
            selected_indices=rollout_indices,
            success_override=success_override,
            score_override=(0.0 if has_human else None),
        )

    done_flag_path = done_flag_path_for(output_base_dir, category_key,
                                        base_episode_name)
    done_flag_path.parent.mkdir(parents=True, exist_ok=True)
    with done_flag_path.open('w', encoding='utf-8') as f:
        f.write('OK\n')

    del df
    gc.collect()


def process_dagger_episodes_to_parquet(
    input_base_dir: Path,
    output_base_dir: Path,
) -> None:
    """Batch-split discovered episodes into human/rollout parquet data."""
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
        process_episode_to_parquet_segments(
            parquet_path,
            sensor_index_path,
            output_base_dir,
            category_key=category_key,
        )


def main():
    parser = argparse.ArgumentParser(
        description=('Split synced parquet episodes into human, '
                     'rollout-success, and rollout-fail parquet subsets.'))
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

    args = parser.parse_args()

    input_base = Path(args.input_base)
    output_base = Path(args.output_base)

    if not input_base.exists():
        print(f'Error: input directory does not exist: {input_base}')
        return

    output_base.mkdir(parents=True, exist_ok=True)

    print(f'Input directory: {input_base}')
    print(f'Output directory: {output_base}')

    process_dagger_episodes_to_parquet(input_base, output_base)
    print('All episodes processed.')


if __name__ == '__main__':
    main()
