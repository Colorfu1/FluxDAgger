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
"""episode_utils.py.

Shared episode helpers for parquet post-processing scripts.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

__all__ = [
    'load_sensor_index',
    'get_episode_items',
    'split_frames_by_mode',
    'crop_sensor_index',
    'normalize_episode_name',
    'category_key_from_path',
    'done_flag_path_for',
]


def load_sensor_index(sensor_index_path: Path) -> Dict:
    """Load ``sensor_index.json`` as a dictionary."""
    with sensor_index_path.open('r', encoding='utf-8') as f:
        return json.load(f)


def normalize_episode_name(parquet_path: Path) -> str:
    """Normalize names such as ``episode_XXX_synced`` to ``episode_XXX``."""
    name = parquet_path.stem.replace('_synced', '')
    if not name.startswith('episode_'):
        name = f'episode_{name}'
    return name


def get_episode_items(input_base_dir: Path) -> List[Tuple[str, Path, Path]]:
    """Recursively discover ``episode_XXX.parquet`` files.

    Returns:
        ``[(episode_name, parquet_path, sensor_index_path), ...]``.
    """
    items: List[Tuple[str, Path, Path]] = []
    seen: set = set()
    for parquet_path in sorted(input_base_dir.rglob('episode_*.parquet')):
        if not parquet_path.is_file():
            continue
        episode_name = parquet_path.stem.replace('_synced', '')
        if not episode_name.startswith('episode_'):
            continue
        sensor_index_path = parquet_path.parent / 'sensor_index.json'
        if not sensor_index_path.exists():
            continue
        key = (episode_name, str(parquet_path))
        if key in seen:
            continue
        seen.add(key)
        items.append((episode_name, parquet_path, sensor_index_path))
    return items


def split_frames_by_mode(
    sensor_data: Dict,
    frame_count: int,
) -> Tuple[List[Dict], bool]:
    """Split a trajectory into contiguous human/rollout segments.

    Returns:
        ``(segments, has_human)``.
    """
    frame_mode_mapping: Dict[str, str] = (
        sensor_data.get('frame_mode_mapping', {}) or {})

    segments: List[Dict] = []
    current_kind: Optional[str] = None
    current_indices: List[int] = []

    for i in range(frame_count):
        mode = frame_mode_mapping.get(str(i), 'rollout')
        kind = 'human' if mode == 'human' else 'rollout'

        if current_kind is None:
            current_kind = kind

        if kind != current_kind:
            if current_indices:
                segments.append({
                    'kind': current_kind,
                    'indices': current_indices,
                })
            current_indices = []
            current_kind = kind

        current_indices.append(i)

    if current_indices:
        segments.append({
            'kind': current_kind,
            'indices': current_indices,
        })

    has_human = any(seg['kind'] == 'human' for seg in segments)
    return segments, has_human


def crop_sensor_index(
    sensor_data: Dict,
    selected_indices: List[int],
    segment_len: int,
    success_override: Optional[bool] = None,
    score_override: Optional[float] = None,
) -> Dict:
    """Crop ``sensor_index`` by selected frame indices and remap ids."""
    new_data: Dict = json.loads(json.dumps(sensor_data))

    id_map: Dict[int, int] = {
        int(old_id): int(new_id)
        for new_id, old_id in enumerate(selected_indices)
    }

    new_data['frame_count'] = segment_len
    if success_override is not None:
        new_data['success'] = bool(success_override)
    if score_override is not None:
        new_data['score'] = float(score_override)

    frame_mode_mapping: Dict[str, str] = (
        new_data.get('frame_mode_mapping', {}) or {})
    new_frame_mode_mapping: Dict[str, str] = {}
    for old_id_str, mode in frame_mode_mapping.items():
        try:
            old_id = int(old_id_str)
        except ValueError:
            continue
        if old_id in id_map:
            new_frame_mode_mapping[str(id_map[old_id])] = mode
    new_data['frame_mode_mapping'] = new_frame_mode_mapping

    sensor_index: Dict[str, List[Dict]] = (
        new_data.get('sensor_index', {}) or {})
    new_sensor_index: Dict[str, List[Dict]] = {}
    for sensor_name, entries in sensor_index.items():
        new_entries: List[Dict] = []
        for entry in entries:
            old_id = entry.get('frame_id')
            if old_id is None or old_id not in id_map:
                continue
            new_entry = dict(entry)
            new_entry['frame_id'] = id_map[old_id]
            if 'synced_frame_id' in new_entry:
                new_entry['synced_frame_id'] = id_map[old_id]
            new_entries.append(new_entry)
        if new_entries:
            new_entries.sort(key=lambda x: x.get('frame_id', 0))
            new_sensor_index[sensor_name] = new_entries
    new_data['sensor_index'] = new_sensor_index

    frame_timestamps = new_data.get('frame_timestamps', [])
    if frame_timestamps:
        new_frame_timestamps: List[Dict] = []
        for item in frame_timestamps:
            old_frame_id = item.get('frame_id')
            if old_frame_id is None or old_frame_id not in id_map:
                continue
            new_item = dict(item)
            new_item['frame_id'] = id_map[old_frame_id]
            new_frame_timestamps.append(new_item)
        new_frame_timestamps.sort(key=lambda x: x.get('frame_id', 0))
        new_data['frame_timestamps'] = new_frame_timestamps

    raw_action_chunks = new_data.get('raw_action_chunks', [])
    if raw_action_chunks:
        new_raw_chunks: List[Dict] = []
        for chunk in raw_action_chunks:
            old_frame_id = chunk.get('frame_id')
            if old_frame_id is None or old_frame_id not in id_map:
                continue
            new_chunk = dict(chunk)
            new_chunk['frame_id'] = id_map[old_frame_id]
            new_raw_chunks.append(new_chunk)
        new_raw_chunks.sort(key=lambda x: x.get('frame_id', 0))
        new_data['raw_action_chunks'] = new_raw_chunks

    return new_data


def category_key_from_path(
    input_base_dir: Path,
    parquet_path: Path,
) -> str:
    """Infer a category key from a path relative to the input root."""
    try:
        rel_parent = parquet_path.parent.relative_to(input_base_dir)
    except ValueError:
        return 'root'
    parts = rel_parent.parts
    category_parts = parts[:-1] if len(parts) > 1 else parts
    return '__'.join(category_parts) if category_parts else 'root'


def done_flag_path_for(
    output_base_dir: Path,
    category_key: str,
    episode_name: str,
) -> Path:
    """Return the per-category done-flag path for an episode."""
    return (output_base_dir / '.done' / f'{category_key}__{episode_name}.done')
