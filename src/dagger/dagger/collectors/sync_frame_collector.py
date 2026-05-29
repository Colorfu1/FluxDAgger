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

from __future__ import annotations
import gc
import json
import os
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from ..infra.dagger_log import get_logger
from ..runtime_config import as_str, load_runtime_config

logger = get_logger(__name__, prefix='SyncFrameDataCollector')


class SyncFrameDataCollector:
    """Collect synchronized frames and write parquet episodes.

    The collector receives already-aligned frames, buffers them in memory, and
    writes a parquet episode when ``finish_episode`` is called.

    Output layout::

        base_dir/scene_dir/YYYYMMDD/checkpoint_folder_name/task_id/episode_XXX/

    Success/failure state is stored in ``sensor_index.json`` instead of encoded
    in subdirectory names.
    """

    def __init__(
        self,
        base_dir=None,
        scene_dir=None,
        checkpoint_dir=None,
        task_id=None,
        default_camera_names=None,
    ):
        """
        Args:
            base_dir: Root output directory.
            scene_dir: Scene name, for example ``fold_clothes``.
            checkpoint_dir: Checkpoint directory.
            task_id: Policy/checkpoint identifier.
        """
        if base_dir is None or default_camera_names is None:
            config = load_runtime_config()
            base_dir = base_dir or as_str(config.paths.data_buffer_dir)
            default_camera_names = (
                default_camera_names or config.default_saved_camera_names())
        self.default_camera_names = list(default_camera_names)
        self.base_dir = base_dir
        self.scene_dir = scene_dir
        self.checkpoint_dir = checkpoint_dir
        self.task_id = task_id

        if checkpoint_dir:
            self.checkpoint_folder_name = os.path.basename(
                checkpoint_dir.rstrip('/'))
        else:
            self.checkpoint_folder_name = None

        self.date_str = datetime.now().strftime('%Y%m%d')

        self.episode_id = 0
        self.step_id = 0
        self.current_trajectory_type = None  # 'rollout', 'human', 'pad'
        self.is_human_mode = False

        self.raw_action_chunks = []

        # {'qpos': np.array, 'qvel': np.array,
        #  'images': {cam_name: np.array}, ...}
        self.episode_buffer = []
        self.episode_buffer_lock = threading.Lock()

        self.current_episode_dir = None

        logger.info('Initialized.')
        logger.info('  Base dir: %s', self.base_dir)
        logger.info('  Scene dir: %s', self.scene_dir)
        logger.info('  Checkpoint folder: %s', self.checkpoint_folder_name)
        logger.info('  Task ID: %s', self.task_id)
        logger.info('  Save format: parquet')

    def _get_base_path(self):
        """Return ``base_dir/scene/date/checkpoint/task``."""
        if not all([
                self.scene_dir,
                self.checkpoint_folder_name,
                self.task_id,
        ]):
            raise ValueError(
                'scene_dir, checkpoint_dir, and task_id must be set '
                'before saving data')

        return os.path.join(
            self.base_dir,
            self.scene_dir,
            self.date_str,
            self.checkpoint_folder_name,
            self.task_id,
        )

    def _find_max_episode_id(self, target_path):
        """Return the largest existing episode id under ``target_path``.

        Args:
            target_path: Directory to scan.

        Returns:
            Largest id, or ``-1`` when no episode exists.
        """
        if not os.path.exists(target_path):
            return -1

        max_id = -1

        for item in os.listdir(target_path):
            item_path = os.path.join(target_path, item)
            if os.path.isdir(item_path) and item.startswith('episode_'):
                episode_num = int(item.split('_')[1])
                max_id = max(max_id, episode_num)

        return max_id

    def start_new_episode(self, trajectory_type=None):
        """Start a new episode directory and clear the in-memory buffer.

        Args:
            trajectory_type: ``rollout`` or ``human``. If ``None``, infer from
                ``is_human_mode``.
        """
        if trajectory_type is None:
            trajectory_type = 'human' if self.is_human_mode else 'rollout'

        base_path = self._get_base_path()
        max_episode_id = self._find_max_episode_id(base_path)
        self.episode_id = max_episode_id + 1

        episode_dir_name = f'episode_{self.episode_id:03d}'
        episode_path = os.path.join(base_path, episode_dir_name)
        while os.path.exists(episode_path):
            logger.warning(
                'Episode %03d already exists in %s,'
                ' using next available ID', self.episode_id, base_path)
            self.episode_id += 1
            episode_dir_name = f'episode_{self.episode_id:03d}'
            episode_path = os.path.join(base_path, episode_dir_name)

        self.current_episode_dir = episode_path
        os.makedirs(self.current_episode_dir, exist_ok=True)

        self.raw_action_chunks = []
        self.step_id = 0
        self.current_trajectory_type = trajectory_type
        current_episode_id = self.episode_id

        with self.episode_buffer_lock:
            self.episode_buffer = []

        logger.info('Started new episode %03d', current_episode_id)
        logger.info('  Trajectory type: %s', trajectory_type)
        logger.info('  Save format: parquet')
        logger.info('  Save path: %s', self.current_episode_dir)

    def set_human_mode(self, enabled=True):
        """Update whether newly started episodes default to human mode."""
        self.is_human_mode = enabled
        if enabled:
            logger.info('Human teleoperation mode enabled')
        else:
            logger.info('Human teleoperation mode disabled')

    @staticmethod
    def _encode_image(img):
        """Encode an image as PNG bytes for parquet storage.

        Args:
            img: BGR numpy image.

        Returns:
            Encoded PNG bytes, or ``None`` for missing images.
        """
        if img is None:
            return None
        _, buf = cv2.imencode('.png', img)
        return buf.tobytes()

    @staticmethod
    def _msg_to_array(msg):
        """Convert JointState / JointTrajectory to a numpy array."""
        if msg is None:
            return None
        if hasattr(msg, 'points'):
            if len(msg.points) == 0:
                return None
            return np.array([p.positions for p in msg.points],
                            dtype=np.float32)
        if hasattr(msg, 'position'):
            return np.array(msg.position, dtype=np.float32)
        return None

    def save_frame_from_synced(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        effort: np.ndarray,
        eepose: np.ndarray,
        action: np.ndarray,
        images: dict[str, np.ndarray],
        timestamps: dict[str, float],
        camera_names: list[str] | None = None,
    ) -> None:
        """Append one synchronized frame to the in-memory episode buffer.

        The upstream sync node has already aligned sensor timestamps; this
        method only stores the aligned numeric arrays, images, and timestamp
        metadata.

        Args:
            qpos: Robot joint positions.
            qvel: Robot joint velocities.
            effort: Robot joint efforts.
            eepose: End-effector pose vector.
            action: Action executed at this frame.
            images: Aligned camera images keyed by camera name.
            timestamps: Timestamp metadata used to build ``sensor_index``.
            camera_names: Camera names to persist. ``None`` uses defaults.

        Returns:
            None. If no episode is active, the frame is skipped with a warning.
        """
        if self.current_episode_dir is None:
            logger.warning(
                'skip save_frame_from_synced: '
                'episode_dir is None (step_id=%d)', self.step_id)
            return

        data_flag = 'human' if self.is_human_mode else 'rollout'
        buffer_frame = {
            'frame_id': self.step_id,
            'data_flag': data_flag,
            'timestamps': timestamps.copy() if timestamps else {},
        }

        buffer_frame['qpos'] = np.asarray(qpos, dtype=np.float32)
        buffer_frame['qvel'] = np.asarray(qvel, dtype=np.float32)
        buffer_frame['effort'] = np.asarray(effort, dtype=np.float32)
        buffer_frame['eepose'] = np.asarray(eepose, dtype=np.float32)

        buffer_frame['action'] = np.asarray(action, dtype=np.float32)

        buffer_frame['images'] = {}
        if camera_names is None:
            camera_names = list(images.keys())

        for cam_name in camera_names:
            img = images.get(cam_name, None)
            if img is not None:
                buffer_frame['images'][cam_name] = img.copy()

        with self.episode_buffer_lock:
            self.episode_buffer.append(buffer_frame)
            self.step_id += 1

    def _save_data_parquet(self,
                           dataset_path,
                           camera_names=None,
                           success=None,
                           score=None):
        """Write ``episode_buffer`` to a limx_aloha-compatible parquet file.

        Args:
            dataset_path: Output path without extension.
            camera_names: Cameras to save; defaults to configured cameras.
            success: Episode success flag.
            score: Score used only for successful episodes.
        """
        t0 = time.time()

        if camera_names is None:
            camera_names = list(self.default_camera_names)

        with self.episode_buffer_lock:
            buffer_copy = list(self.episode_buffer)

        if len(buffer_copy) == 0:
            logger.warning('No data to save')
            return

        data_dict = {
            '/observations/qpos': [],
            '/observations/qvel': [],
            '/observations/effort': [],
            '/observations/eepose': [],
            '/action': [],
            '/data_flag': [],
            '/success': [],
            '/score': [],
        }

        success_int = 1 if success else 0
        score_int = score if (success and score is not None) else -1

        for cam_name in camera_names:
            data_dict[f'/observations/images/{cam_name}'] = []

        with ThreadPoolExecutor() as executor:
            image_tasks = []
            task_map = []

            for frame_idx, frame_data in enumerate(buffer_copy):
                qpos = frame_data.get('qpos')
                qvel = frame_data.get('qvel')
                effort = frame_data.get('effort')
                eepose = frame_data.get('eepose')
                action = frame_data.get('action')
                data_flag_str = frame_data.get('data_flag', 'rollout')
                data_flag_int = 1 if data_flag_str == 'human' else 0

                data_dict['/observations/qpos'].append(
                    qpos if qpos is not None else np.
                    zeros(14, dtype=np.float32))
                data_dict['/observations/qvel'].append(
                    qvel if qvel is not None else np.
                    zeros(14, dtype=np.float32))
                data_dict['/observations/effort'].append(
                    effort if effort is not None else np.
                    zeros(14, dtype=np.float32))
                data_dict['/observations/eepose'].append(
                    eepose if eepose is not None else np.
                    zeros(14, dtype=np.float32))
                data_dict['/action'].append(
                    action if action is not None else np.
                    zeros(14, dtype=np.float32))
                data_dict['/data_flag'].append(data_flag_int)
                data_dict['/success'].append(success_int)
                data_dict['/score'].append(score_int)

                images = frame_data.get('images', {})
                for cam_name in camera_names:
                    img = images.get(cam_name)
                    if img is not None:
                        task = executor.submit(self._encode_image, img)
                        image_tasks.append(task)
                        task_map.append((frame_idx, cam_name))
                    else:
                        image_tasks.append(None)
                        task_map.append((frame_idx, cam_name))

            num_frames = len(buffer_copy)
            image_data = {
                cam_name: [None] * num_frames
                for cam_name in camera_names
            }

            for task, (frame_idx, cam_name) in zip(image_tasks, task_map):
                if task is not None:
                    encoded_img = task.result()
                    image_data[cam_name][frame_idx] = encoded_img

            for cam_name in camera_names:
                data_dict[f'/observations/images/{cam_name}'] = image_data[
                    cam_name]

        df = pd.DataFrame(data_dict)

        pq_path = dataset_path + '.parquet'
        df.to_parquet(
            pq_path, engine='pyarrow', compression='snappy', index=False)

        t1 = time.time()
        num_frames = len(buffer_copy)

        for key in list(data_dict.keys()):
            if '/images/' in key:
                data_dict[key] = None
        del data_dict

        for cam_name in list(image_data.keys()):
            image_data[cam_name] = None
        del image_data

        del df

        for frame_data in buffer_copy:
            if 'images' in frame_data and frame_data['images'] is not None:
                for cam_name in list(frame_data['images'].keys()):
                    frame_data['images'][cam_name] = None
                frame_data['images'] = None
        del buffer_copy

        gc.collect()

        logger.info_green('Parquet saved: %s', pq_path)
        logger.info('  Frames: %d, Time: %.2fs', num_frames, t1 - t0)

    def record_raw_action_chunk(self, left_msg, right_msg, frame_id=None):
        """Record raw action chunks outside the strict 30Hz sync path.

        Args:
            left_msg, right_msg: Usually ``JointTrajectory`` or ``JointState``.
            frame_id: Optional synced frame id used in output filenames.
        """
        if left_msg is None and right_msg is None:
            return

        ts = None
        if left_msg is not None and hasattr(left_msg, 'header'):
            ts = left_msg.header.stamp.to_sec()
        elif right_msg is not None and hasattr(right_msg, 'header'):
            ts = right_msg.header.stamp.to_sec()

        if ts is None or self.current_episode_dir is None:
            return

        left_arr = self._msg_to_array(left_msg)
        right_arr = self._msg_to_array(right_msg)

        episode_dir = Path(self.current_episode_dir)
        actions_dir = episode_dir / 'actions'
        actions_dir.mkdir(parents=True, exist_ok=True)

        left_file = None
        right_file = None

        if frame_id is not None:
            if left_arr is not None:
                left_file = actions_dir / f'left_raw_chunk_{frame_id}.npy'
                np.save(left_file, left_arr)
            if right_arr is not None:
                right_file = actions_dir / f'right_raw_chunk_{frame_id}.npy'
                np.save(right_file, right_arr)
        else:
            ts_str = f'{ts:.6f}'.replace('.', '_')
            if left_arr is not None:
                left_file = (
                    actions_dir / f'puppet_arm_left_raw_chunk_{ts_str}.npy')
                np.save(left_file, left_arr)
            if right_arr is not None:
                right_file = (
                    actions_dir / f'puppet_arm_right_raw_chunk_{ts_str}.npy')
                np.save(right_file, right_arr)

        self.raw_action_chunks.append({
            'frame_id':
            frame_id,
            'timestamp':
            float(ts),
            'left_file':
            str(left_file) if left_file is not None else None,
            'right_file':
            str(right_file) if right_file is not None else None,
        })

    def discard_episode(self):
        """Drop the current episode and delete its output directory."""
        if self.current_episode_dir is None:
            logger.warning('No episode to discard.')
            return

        episode_dir = self.current_episode_dir
        logger.warning('Discarding episode: %s', episode_dir)

        with self.episode_buffer_lock:
            for frame_data in self.episode_buffer:
                if 'images' in frame_data and frame_data['images'] is not None:
                    for cam_name in list(frame_data['images'].keys()):
                        frame_data['images'][cam_name] = None
                    frame_data['images'] = None
            self.episode_buffer.clear()
            self.episode_buffer = []

        self.raw_action_chunks = []
        self.current_episode_dir = None
        self.step_id = 0

        if os.path.exists(episode_dir):
            try:
                shutil.rmtree(episode_dir)
                logger.warning('Deleted directory: %s', episode_dir)
            except OSError as exc:
                logger.warning('Failed to delete %s: %s', episode_dir, exc)

        gc.collect()
        logger.info('Episode discarded, memory released')

    def finish_episode(self,
                       success=None,
                       score=None,
                       trajectory_type=None,
                       camera_names=None):
        """
        Finish the current episode and write it to parquet.

        Args:
            success: Episode success flag. ``None`` is treated as failure.
            score: Score used only for successful episodes.
            trajectory_type: ``rollout`` or ``human``.
            camera_names: Cameras to save.
        """
        if self.current_episode_dir is None:
            logger.warning('No episode to finish.')
            return

        with self.episode_buffer_lock:
            has_data = len(self.episode_buffer) > 0

        if not has_data:
            logger.warning('No data collected in this episode.')
            self.raw_action_chunks = []
            with self.episode_buffer_lock:
                self.episode_buffer = []
            self.current_episode_dir = None
            self.step_id = 0
            return

        current_episode_id = None
        if self.current_episode_dir:
            dir_name = os.path.basename(self.current_episode_dir)
            if dir_name.startswith('episode_'):
                current_episode_id = int(dir_name.split('_')[1])
            else:
                current_episode_id = self.episode_id - 1
        else:
            current_episode_id = self.episode_id - 1

        if success is None:
            logger.warning(
                'success is None, defaulting to '
                'failure. Please ensure success is provided by the caller.')
            success = False

        if trajectory_type is None:
            trajectory_type = self.current_trajectory_type

        self._finish_episode_parquet(current_episode_id, success, score,
                                     trajectory_type, camera_names)

    def _finish_episode_parquet(self,
                                episode_id,
                                success,
                                score,
                                trajectory_type,
                                camera_names=None):
        """Write episode parquet and sidecar metadata."""
        episode_dir_name = os.path.basename(self.current_episode_dir)
        dataset_path = os.path.join(self.current_episode_dir, episode_dir_name)
        self._save_data_parquet(dataset_path, camera_names, success, score)

        frame_mode_mapping = {}
        frame_timestamps_info = []
        with self.episode_buffer_lock:
            for frame_data in self.episode_buffer:
                frame_id = frame_data.get('frame_id')
                data_flag = frame_data.get('data_flag', 'rollout')
                if frame_id is not None:
                    frame_mode_mapping[str(frame_id)] = data_flag

                timestamps = frame_data.get('timestamps', {})
                sync_time = timestamps.get('sync_time', None)

                if sync_time is not None:
                    frame_ts_info = {
                        'frame_id': frame_id,
                        'sync_time': sync_time,
                        'sensor_timestamps': {},
                        'sensor_diffs': {},
                    }

                    for sensor_name, sensor_ts in timestamps.items():
                        if sensor_name == 'sync_time':
                            continue
                        if sensor_ts is not None:
                            frame_ts_info['sensor_timestamps'][
                                sensor_name] = sensor_ts
                            frame_ts_info['sensor_diffs'][
                                sensor_name] = sensor_ts - sync_time

                    frame_timestamps_info.append(frame_ts_info)
            buffer_size = len(self.episode_buffer)

        sync_times = [info['sync_time'] for info in frame_timestamps_info]

        processed_raw_action_chunks = []
        for chunk_info in self.raw_action_chunks:
            chunk_ts = chunk_info.get('timestamp')
            frame_id = chunk_info.get('frame_id')

            if chunk_ts is not None and len(sync_times) > 0:
                min_diff = float('inf')
                nearest_sync_time = None
                for sync_time in sync_times:
                    diff = abs(chunk_ts - sync_time)
                    if diff < min_diff:
                        min_diff = diff
                        nearest_sync_time = sync_time

                diff_from_sync = (
                    chunk_ts - nearest_sync_time
                    if nearest_sync_time is not None else None)

                processed_chunk = {
                    'frame_id': frame_id,
                    'timestamp': chunk_ts,
                    'nearest_sync_time': nearest_sync_time,
                    'diff_from_sync': diff_from_sync,
                    'left_file': chunk_info.get('left_file'),
                    'right_file': chunk_info.get('right_file'),
                }
                processed_raw_action_chunks.append(processed_chunk)
            else:
                processed_raw_action_chunks.append(chunk_info)

        episode_metadata = {
            'episode_id': episode_id,
            'trajectory_type': trajectory_type,
            'success': success,
            'timestamp': datetime.now().isoformat(),
            'collection_mode': 'sync',
            'frame_count': buffer_size,
            'frame_mode_mapping': frame_mode_mapping,
            'raw_action_chunks': processed_raw_action_chunks,
            'frame_timestamps': frame_timestamps_info,
        }

        if success and score is not None:
            episode_metadata['score'] = score

        metadata_path = os.path.join(self.current_episode_dir,
                                     'sensor_index.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(episode_metadata, f, indent=2, ensure_ascii=False)

        status = 'Success' if success else 'Failure'
        logger.info('Finished episode %03d', episode_id)
        logger.info('  Type: %s - %s', status, trajectory_type)
        logger.info('  Frames: %d', buffer_size)
        logger.info('  Metadata saved to: %s', metadata_path)

        with self.episode_buffer_lock:
            for frame_data in self.episode_buffer:
                if 'images' in frame_data and frame_data['images'] is not None:
                    for cam_name in list(frame_data['images'].keys()):
                        frame_data['images'][cam_name] = None
                    frame_data['images'] = None
            self.episode_buffer.clear()
            self.episode_buffer = []

        self.raw_action_chunks = []
        self.current_episode_dir = None
        self.step_id = 0

        gc.collect()
        logger.info('Memory released after episode finish')
