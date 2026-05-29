#!/usr/bin/env python3
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
"""Visualize parquet image streams and numeric time series.

Usage:
    python visualize_parquet.py <parquet_file_path> [options]

Examples:
    python visualize_parquet.py \\
        /home/agilex/data/20251229/\\
fold_fabric_dagger_01_agilex_aloha_02/episode_0.parquet
    python visualize_parquet.py <parquet_file> \\
        --show-images --show-plots --fps 30
    python visualize_parquet.py <parquet_file> \\
        --save-video --save-frames --save-plots --save-qpos --fps 30
"""

from __future__ import annotations
import argparse
import sys
import traceback
from pathlib import Path
from typing import List, Optional

import cv2
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use('Agg', force=True)

chinese_fonts = []
for font in fm.fontManager.ttflist:
    font_name = font.name.lower()
    if any(keyword in font_name for keyword in [
            'simhei',
            'simsun',
            'microsoft yahei',
            'wenquanyi',
            'noto',
            'source han',
    ]):
        chinese_fonts.append(font.name)

if chinese_fonts:
    plt.rcParams['font.sans-serif'] = chinese_fonts + [
        'DejaVu Sans',
        'Arial Unicode MS',
        'sans-serif',
    ]
else:
    plt.rcParams['font.sans-serif'] = [
        'DejaVu Sans',
        'Arial Unicode MS',
        'SimHei',
        'WenQuanYi Micro Hei',
        'sans-serif',
    ]

plt.rcParams['axes.unicode_minus'] = False


def is_zero_padded(data, tolerance=1e-6) -> bool:
    """Return whether numeric data is effectively zero padding."""
    if data is None:
        return True
    try:
        arr = np.array(data)
        if np.allclose(arr, 0.0, atol=tolerance):
            return True
        return False
    except Exception:
        return True


def decode_image(encoded_img) -> Optional[np.ndarray]:
    """Decode a PNG-encoded image."""
    if encoded_img is None:
        return None
    if isinstance(encoded_img, bytes):
        nparr = np.frombuffer(encoded_img, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    return None


def get_image_columns(df: pd.DataFrame) -> List[str]:
    """Return all image column names."""
    image_cols = []
    for col in df.columns:
        if ('/observations/images/' in col
                or '/observations/images_depth/' in col):
            image_cols.append(col)
    return image_cols


def get_numeric_columns(df: pd.DataFrame) -> List[str]:
    """Detect numeric columns automatically.

    Includes numeric, boolean, numeric-array object columns, and data flags.
    """
    numeric_cols = []

    for col in df.columns:
        if ('/observations/images/' in col
                or '/observations/images_depth/' in col):
            continue

        dtype = df[col].dtype

        if pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_bool_dtype(
                dtype):
            numeric_cols.append(col)
        elif (col.endswith('data_flag') or col == '/data_flag'
              or col == 'data_flag'):
            numeric_cols.append(col)
        elif dtype == 'object':
            sample_size = min(10, len(df))
            is_numeric_array = False

            for i in range(sample_size):
                val = df[col].iloc[i]
                if val is None:
                    continue
                if isinstance(val, (list, np.ndarray)):
                    try:
                        arr = np.array(val)
                        if arr.size > 0 and pd.api.types.is_numeric_dtype(
                                arr.dtype):
                            is_numeric_array = True
                            break
                    except Exception:
                        pass

            if is_numeric_array:
                numeric_cols.append(col)

    return numeric_cols


def save_images_video(
    df: pd.DataFrame,
    image_cols: List[str],
    output_dir: Path,
    fps: float = 30.0,
):
    """Save image sequences as video files."""
    if not image_cols:
        print('No image columns found')
        return

    print(f'\nFound {len(image_cols)} image columns: {image_cols}')
    output_dir.mkdir(parents=True, exist_ok=True)

    for col in image_cols:
        cam_name = col.split('/')[-1]
        video_path = output_dir / f'{cam_name}.mp4'

        first_img = None
        for frame_idx in range(len(df)):
            img = decode_image(df[col].iloc[frame_idx])
            if img is not None:
                first_img = img
                break

        if first_img is None:
            print(f'  Warning: {cam_name} has no valid images; skipped')
            continue

        height, width = first_img.shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

        frame_count = 0
        for frame_idx in range(len(df)):
            img = decode_image(df[col].iloc[frame_idx])
            if img is not None:
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                out.write(img_bgr)
                frame_count += 1

        out.release()
        print(f'  Saved {cam_name}: {video_path} '
              f'({frame_count} frames, {fps} FPS)')

    if len(image_cols) > 1:
        first_images = {}
        for col in image_cols:
            for frame_idx in range(len(df)):
                img = decode_image(df[col].iloc[frame_idx])
                if img is not None:
                    first_images[col] = img
                    break

        if len(first_images) > 1:
            num_cams = len(first_images)
            if num_cams == 2:
                rows, cols = 1, 2
            elif num_cams <= 4:
                rows, cols = 2, 2
            else:
                rows = (num_cams + 3) // 4
                cols = 4

            cell_h = min(img.shape[0] for img in first_images.values())
            cell_w = min(img.shape[1] for img in first_images.values())

            merged_video_path = output_dir / 'merged_all_cameras.mp4'
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(
                str(merged_video_path),
                fourcc,
                fps,
                (cell_w * cols, cell_h * rows),
            )

            for frame_idx in range(len(df)):
                grid_img = np.zeros((cell_h * rows, cell_w * cols, 3),
                                    dtype=np.uint8)

                for idx, col in enumerate(image_cols):
                    if col not in first_images:
                        continue
                    img = decode_image(df[col].iloc[frame_idx])
                    if img is None:
                        continue
                    if (img.shape[0] != cell_h or img.shape[1] != cell_w):
                        img = cv2.resize(
                            img, (cell_w, cell_h),
                            interpolation=cv2.INTER_AREA)
                    row_idx = idx // cols
                    col_idx = idx % cols
                    grid_img[row_idx * cell_h:(row_idx + 1) * cell_h,
                             col_idx * cell_w:(col_idx + 1) * cell_w, ] = img

                grid_bgr = cv2.cvtColor(grid_img, cv2.COLOR_RGB2BGR)
                out.write(grid_bgr)

            out.release()
            print(f'  Saved merged video: {merged_video_path}')


def save_images_frames(df: pd.DataFrame, image_cols: List[str],
                       output_dir: Path):
    """Save image sequences as individual frame files."""
    if not image_cols:
        print('No image columns found')
        return

    print(f'\nFound {len(image_cols)} image columns: {image_cols}')
    output_dir.mkdir(parents=True, exist_ok=True)

    for col in image_cols:
        cam_name = col.split('/')[-1]
        cam_dir = output_dir / cam_name
        cam_dir.mkdir(exist_ok=True)

        frame_count = 0
        for frame_idx in range(len(df)):
            img = decode_image(df[col].iloc[frame_idx])
            if img is not None:
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                save_path = cam_dir / f'frame_{frame_idx:06d}.png'
                cv2.imwrite(str(save_path), img_bgr)
                frame_count += 1

        print(f'  Saved {cam_name}: {cam_dir} ({frame_count} frames)')


def save_numeric_data_plots(
    df: pd.DataFrame,
    numeric_cols: List[str],
    output_path: Path,
    filter_zero_padding: bool = True,
):
    """Save numeric time-series plots.

    Args:
        df: Source DataFrame.
        numeric_cols: Numeric columns to plot.
        output_path: Output image path.
        filter_zero_padding: Whether to mask zero-padded values.
    """
    if not numeric_cols:
        print('No numeric columns found')
        return

    print(f'\nFound {len(numeric_cols)} numeric columns: {numeric_cols}')
    if len(numeric_cols) == 0:
        print('  Warning: no numeric columns found; check data format')

    print('\nDebug info - all non-image columns:')
    for col in df.columns:
        if ('/observations/images/' not in col
                and '/observations/images_depth/' not in col):
            dtype = df[col].dtype
            sample_val = df[col].iloc[0] if len(df) > 0 else None
            is_included = col in numeric_cols
            print(f'  {col:40s} dtype={str(dtype):15s} '
                  f'sample={str(sample_val)[:50]:50s} '
                  f'included={is_included}')

    num_cols = len(numeric_cols)
    if num_cols == 0:
        return

    rows = (num_cols + 1) // 2
    cols = 2 if num_cols > 1 else 1
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    if num_cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if rows > 1 else axes

    for idx, col in enumerate(numeric_cols):
        ax = axes[idx]
        data_list = df[col].tolist()

        try:
            processed_data = []
            for d in data_list:
                if d is None:
                    processed_data.append(np.nan)
                elif (col.endswith('data_flag') or col == '/data_flag'
                      or col == 'data_flag') and isinstance(d, str):
                    if d.lower() == 'human':
                        processed_data.append(1.0)
                    elif d.lower() == 'rollout':
                        processed_data.append(0.0)
                    else:
                        processed_data.append(np.nan)
                elif isinstance(d, (list, np.ndarray)):
                    if len(d) == 0:
                        processed_data.append(np.nan)
                    else:
                        processed_data.append(np.array(d))
                else:
                    if pd.isna(d):
                        processed_data.append(np.nan)
                    else:
                        processed_data.append(d)

            try:
                data_array = np.array(processed_data)
            except (ValueError, TypeError):
                max_len = 0
                for d in processed_data:
                    if isinstance(d, np.ndarray) and not np.isnan(d).all():
                        max_len = max(max_len, len(d))

                if max_len > 0:
                    data_array = np.full((len(processed_data), max_len),
                                         np.nan)
                    for i, d in enumerate(processed_data):
                        if isinstance(d, np.ndarray) and len(d) > 0:
                            data_array[i, :len(d)] = d
                        elif not (isinstance(d, (float, np.floating))
                                  and np.isnan(d)):
                            if max_len == 1:
                                data_array[i, 0] = d
                else:
                    ax.text(
                        0.5,
                        0.5,
                        'Unsupported data format',
                        transform=ax.transAxes,
                        ha='center',
                        va='center',
                    )
                    ax.set_title(col.split('/')[-1])
                    continue

            if data_array.ndim == 1:
                plot_data = data_array.copy()
                if filter_zero_padding and not (col.endswith('data_flag')
                                                or col == '/data_flag'
                                                or col == 'data_flag'):
                    for i in range(len(plot_data)):
                        if not np.isnan(plot_data[i]) and i < len(data_list):
                            val = data_list[i]
                            if is_zero_padded(val):
                                plot_data[i] = np.nan
                ax.plot(
                    plot_data,
                    label=col.split('/')[-1],
                    marker='.',
                    markersize=1,
                )
                ax.set_ylabel(col.split('/')[-1])
            elif data_array.ndim == 2:
                dim = data_array.shape[1]

                if filter_zero_padding:
                    is_action_col = col == '/action' or col.endswith('/action')

                    if is_action_col and dim >= 14:
                        for i in range(len(data_list)):
                            val = data_list[i]
                            if val is None:
                                if i < data_array.shape[0]:
                                    data_array[i, :] = np.nan
                                continue

                            try:
                                arr = np.array(val)
                            except Exception:
                                if i < data_array.shape[0]:
                                    data_array[i, :] = np.nan
                                continue

                            left_arm = arr[:7]
                            right_arm = arr[7:14]

                            if is_zero_padded(left_arm):
                                if i < data_array.shape[0]:
                                    data_array[i, :7] = np.nan

                            if is_zero_padded(right_arm):
                                if i < data_array.shape[0]:
                                    data_array[i, 7:14] = np.nan
                    else:
                        zero_padded_frames = set()
                        for i in range(len(data_list)):
                            val = data_list[i]
                            if is_zero_padded(val):
                                zero_padded_frames.add(i)

                        for frame_idx in zero_padded_frames:
                            if frame_idx < data_array.shape[0]:
                                data_array[frame_idx, :] = np.nan

                if dim <= 14:
                    for d in range(dim):
                        plot_data = data_array[:, d].copy()
                        ax.plot(
                            plot_data,
                            label=f'dim{d}',
                            alpha=0.7,
                            marker='.',
                            markersize=1,
                        )
                else:
                    for d in range(min(7, dim)):
                        plot_data = data_array[:, d].copy()
                        ax.plot(
                            plot_data,
                            label=f'dim{d}',
                            alpha=0.7,
                            marker='.',
                            markersize=1,
                        )
                    ax.text(
                        0.02,
                        0.98,
                        f'... (showing 7/{dim} dims)',
                        transform=ax.transAxes,
                        verticalalignment='top',
                        bbox=dict(
                            boxstyle='round', facecolor='wheat', alpha=0.5),
                    )

                ax.set_ylabel(col.split('/')[-1])
                ax.legend(loc='upper right', fontsize=8, ncol=2)
            else:
                ax.text(
                    0.5,
                    0.5,
                    f'Unsupported shape: {data_array.shape}',
                    transform=ax.transAxes,
                    ha='center',
                    va='center',
                )

            ax.set_xlabel('Frame')
            ax.set_title(col.split('/')[-1])
            ax.grid(True, alpha=0.3)
        except Exception as e:
            ax.text(
                0.5,
                0.5,
                f'Error: {str(e)[:50]}',
                transform=ax.transAxes,
                ha='center',
                va='center',
            )
            ax.set_title(col.split('/')[-1])

    for idx in range(len(numeric_cols), len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.suptitle(
        f'Numeric Data Time Series (Total: {len(df)} frames)',
        y=0.98,
        fontsize=14,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved numeric data plot: {output_path}')


def save_qpos_plots(df: pd.DataFrame,
                    output_path: Path,
                    filter_zero_padding: bool = True):
    """Save qpos joint-by-joint plots."""
    qpos_col = '/observations/qpos'

    if qpos_col not in df.columns:
        print('No qpos column found')
        return

    print('\nGenerating qpos joint-by-joint plots...')
    data_list = df[qpos_col].tolist()

    try:
        processed_data = []
        for d in data_list:
            if d is None:
                processed_data.append(np.nan)
            elif isinstance(d, (list, np.ndarray)):
                if len(d) == 0:
                    processed_data.append(np.nan)
                else:
                    processed_data.append(np.array(d))
            else:
                if pd.isna(d):
                    processed_data.append(np.nan)
                else:
                    processed_data.append(d)

        try:
            data_array = np.array(processed_data)
        except (ValueError, TypeError):
            max_len = 0
            for d in processed_data:
                if isinstance(d, np.ndarray) and not np.isnan(d).all():
                    max_len = max(max_len, len(d))

            if max_len > 0:
                data_array = np.full((len(processed_data), max_len), np.nan)
                for i, d in enumerate(processed_data):
                    if isinstance(d, np.ndarray) and len(d) > 0:
                        data_array[i, :len(d)] = d
            else:
                print('  Error: unsupported qpos data format')
                return

        if data_array.ndim != 2:
            print(f'  Error: invalid qpos data shape: {data_array.shape}')
            return

        num_joints = data_array.shape[1]
        print(f'  qpos dimension: {num_joints} joints')

        if filter_zero_padding:
            zero_padded_frames = set()
            for i in range(len(data_list)):
                val = data_list[i]
                if is_zero_padded(val):
                    zero_padded_frames.add(i)

            for frame_idx in zero_padded_frames:
                if frame_idx < data_array.shape[0]:
                    data_array[frame_idx, :] = np.nan

        cols_per_row = 3
        rows = (num_joints + cols_per_row - 1) // cols_per_row
        fig, axes = plt.subplots(rows, cols_per_row, figsize=(15, 4 * rows))

        if num_joints == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if rows > 1 else axes

        for joint_idx in range(num_joints):
            ax = axes[joint_idx]
            joint_data = data_array[:, joint_idx].copy()

            ax.plot(
                joint_data,
                label=f'Joint {joint_idx}',
                linewidth=1.5,
                marker='.',
                markersize=2,
            )
            ax.set_xlabel('Frame')
            ax.set_ylabel(f'Joint {joint_idx}')
            ax.set_title(f'Joint {joint_idx}')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=8)

        for idx in range(num_joints, len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.suptitle(
            f'QPOS Joint-by-Joint Visualization '
            f'(Total: {len(df)} frames, {num_joints} joints)',
            y=0.98,
            fontsize=14,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  Saved qpos joint overview: {output_path}')

        for joint_idx in range(num_joints):
            joint_data = data_array[:, joint_idx].copy()

            fig_single, ax_single = plt.subplots(figsize=(12, 6))
            ax_single.plot(
                joint_data,
                label=f'Joint {joint_idx}',
                linewidth=1.5,
                marker='.',
                markersize=2,
            )
            ax_single.set_xlabel('Frame', fontsize=12)
            ax_single.set_ylabel(f'Joint {joint_idx} Value', fontsize=12)
            ax_single.set_title(
                f'QPOS Joint {joint_idx} (Total: {len(df)} frames)',
                fontsize=14,
            )
            ax_single.grid(True, alpha=0.3)
            ax_single.legend(loc='upper right', fontsize=10)

            plt.tight_layout()

            joint_output_path = (
                output_path.parent / f'qpos_joint_{joint_idx}.png')
            plt.savefig(str(joint_output_path), dpi=150, bbox_inches='tight')
            plt.close()

        print(f'  Saved {num_joints} qpos per-joint plots')

    except Exception as e:
        print(f'  Error: failed to generate qpos plots: {e}')
        traceback.print_exc()


def save_action_plots(df: pd.DataFrame,
                      output_path: Path,
                      filter_zero_padding: bool = True):
    """Save action joint-by-joint plots."""
    action_col = '/action'

    if action_col not in df.columns:
        print('No action column found')
        return

    print('\nGenerating action joint-by-joint plots...')
    data_list = df[action_col].tolist()

    try:
        processed_data = []
        for d in data_list:
            if d is None:
                processed_data.append(np.nan)
            elif isinstance(d, (list, np.ndarray)):
                if len(d) == 0:
                    processed_data.append(np.nan)
                else:
                    processed_data.append(np.array(d))
            else:
                if pd.isna(d):
                    processed_data.append(np.nan)
                else:
                    processed_data.append(d)

        try:
            data_array = np.array(processed_data)
        except (ValueError, TypeError):
            max_len = 0
            for d in processed_data:
                if isinstance(d, np.ndarray) and not np.isnan(d).all():
                    max_len = max(max_len, len(d))

            if max_len > 0:
                data_array = np.full((len(processed_data), max_len), np.nan)
                for i, d in enumerate(processed_data):
                    if isinstance(d, np.ndarray) and len(d) > 0:
                        data_array[i, :len(d)] = d
            else:
                print('  Error: unsupported action data format')
                return

        if data_array.ndim != 2:
            print(f'  Error: invalid action data shape: {data_array.shape}')
            return

        num_joints = data_array.shape[1]
        print(f'  action dimension: {num_joints} joints')

        if filter_zero_padding:
            for i in range(len(data_list)):
                val = data_list[i]
                if val is None:
                    if i < data_array.shape[0]:
                        data_array[i, :] = np.nan
                    continue

                try:
                    arr = np.array(val)
                except Exception:
                    if i < data_array.shape[0]:
                        data_array[i, :] = np.nan
                    continue

                if len(arr) >= 14:
                    left_arm = arr[:7]
                    right_arm = arr[7:14]

                    if is_zero_padded(left_arm):
                        if i < data_array.shape[0]:
                            data_array[i, :7] = np.nan

                    if is_zero_padded(right_arm):
                        if i < data_array.shape[0]:
                            data_array[i, 7:14] = np.nan
                else:
                    if is_zero_padded(val):
                        if i < data_array.shape[0]:
                            data_array[i, :] = np.nan

        cols_per_row = 3
        rows = (num_joints + cols_per_row - 1) // cols_per_row
        fig, axes = plt.subplots(rows, cols_per_row, figsize=(15, 4 * rows))

        if num_joints == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if rows > 1 else axes

        for joint_idx in range(num_joints):
            ax = axes[joint_idx]
            joint_data = data_array[:, joint_idx].copy()

            ax.plot(
                joint_data,
                label=f'Joint {joint_idx}',
                linewidth=1.5,
                marker='.',
                markersize=2,
            )
            ax.set_xlabel('Frame')
            ax.set_ylabel(f'Joint {joint_idx}')
            ax.set_title(f'Joint {joint_idx}')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=8)

        for idx in range(num_joints, len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.suptitle(
            f'Action Joint-by-Joint Visualization '
            f'(Total: {len(df)} frames, {num_joints} joints)',
            y=0.98,
            fontsize=14,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  Saved action joint overview: {output_path}')

        for joint_idx in range(num_joints):
            joint_data = data_array[:, joint_idx].copy()

            fig_single, ax_single = plt.subplots(figsize=(12, 6))
            ax_single.plot(
                joint_data,
                label=f'Joint {joint_idx}',
                linewidth=1.5,
                marker='.',
                markersize=2,
            )
            ax_single.set_xlabel('Frame', fontsize=12)
            ax_single.set_ylabel(f'Joint {joint_idx} Value', fontsize=12)
            ax_single.set_title(
                f'Action Joint {joint_idx} (Total: {len(df)} frames)',
                fontsize=14,
            )
            ax_single.grid(True, alpha=0.3)
            ax_single.legend(loc='upper right', fontsize=10)

            plt.tight_layout()

            joint_output_path = (
                output_path.parent / f'action_joint_{joint_idx}.png')
            plt.savefig(str(joint_output_path), dpi=150, bbox_inches='tight')
            plt.close()

        print(f'  Saved {num_joints} action per-joint plots')

    except Exception as e:
        print(f'  Error: failed to generate action plots: {e}')
        traceback.print_exc()


def print_data_summary(df: pd.DataFrame):
    """Print a summary of the parquet data."""
    print('\n' + '=' * 80)
    print('Data Summary')
    print('=' * 80)
    print(f'Total frames: {len(df)}')
    print(f'Total columns: {len(df.columns)}')

    print('\nColumns:')
    for i, col in enumerate(df.columns, 1):
        non_null = df[col].notna().sum()
        print(f'  {i:2d}. {col:50s} '
              f'(non-null: {non_null:5d}/{len(df):5d})')

    image_cols = get_image_columns(df)
    if image_cols:
        print(f'\nImage columns ({len(image_cols)}):')
        for col in image_cols:
            cam_name = col.split('/')[-1]
            non_null = df[col].notna().sum()
            print(f'  - {cam_name}: {non_null}/{len(df)} frames')

    numeric_cols = get_numeric_columns(df)
    if numeric_cols:
        print(f'\nNumeric columns ({len(numeric_cols)}):')
        for col in numeric_cols:
            non_null = df[col].notna().sum()
            data_list = df[col].dropna().tolist()
            if data_list:
                try:
                    sample = (
                        np.array(data_list[0]) if isinstance(
                            data_list[0],
                            (list, np.ndarray)) else data_list[0])
                    if isinstance(sample, np.ndarray):
                        col_short = col.split('/')[-1]
                        print(f'  - {col_short}: '
                              f'{non_null}/{len(df)} frames, '
                              f'shape={sample.shape}')
                    else:
                        print(f"  - {col.split('/')[-1]}: "
                              f'{non_null}/{len(df)} frames')
                except Exception:
                    print(f"  - {col.split('/')[-1]}: "
                          f'{non_null}/{len(df)} frames')


def main():
    parser = argparse.ArgumentParser(
        description='Visualize parquet data files.', )
    parser.add_argument('parquet_file', type=str, help='Path to parquet file.')
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory. Defaults to a directory based on the parquet '
        'file name.',
    )
    parser.add_argument(
        '--summary-only',
        action='store_true',
        help='Only print summary information; do not save visualizations.')
    parser.add_argument(
        '--fps', type=float, default=30.0, help='Video FPS. Defaults to 30.0.')
    parser.add_argument(
        '--save-video',
        action='store_true',
        help='Save image sequences as MP4 videos.')
    parser.add_argument(
        '--save-frames',
        action='store_true',
        help='Save image sequences as PNG frame files.')
    parser.add_argument(
        '--save-plots',
        action='store_true',
        help='Save numeric time-series plots.')
    parser.add_argument(
        '--save-qpos',
        action='store_true',
        help='Save qpos joint-by-joint plots.')
    parser.add_argument(
        '--save-action',
        action='store_true',
        help='Save action joint-by-joint plots.')
    parser.add_argument(
        '--no-filter-zero-padding',
        action='store_true',
        help='Do not filter zero-padded data during visualization.',
    )

    args = parser.parse_args()

    parquet_path = Path(args.parquet_file)
    if not parquet_path.exists():
        print(f'Error: file does not exist: {parquet_path}')
        sys.exit(1)

    print(f'Reading parquet file: {parquet_path}')
    print(f'File size: {parquet_path.stat().st_size / (1024**2):.2f} MB')

    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        print(f'Error: failed to read parquet file: {e}')
        sys.exit(1)

    print_data_summary(df)

    if args.summary_only:
        return

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        parquet_stem = parquet_path.stem
        output_dir = parquet_path.parent / f'{parquet_stem}_visualization'

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f'\nOutput directory: {output_dir}')

    if args.save_video:
        image_cols = get_image_columns(df)
        if image_cols:
            video_dir = output_dir / 'videos'
            save_images_video(df, image_cols, video_dir, fps=args.fps)
        else:
            print('No image columns found; skipping video export')

    if args.save_frames:
        image_cols = get_image_columns(df)
        if image_cols:
            frames_dir = output_dir / 'frames'
            save_images_frames(df, image_cols, frames_dir)
        else:
            print('No image columns found; skipping frame export')

    if args.save_plots:
        numeric_cols = get_numeric_columns(df)
        if numeric_cols:
            plots_path = output_dir / 'numeric_data_plots.png'
            save_numeric_data_plots(
                df,
                numeric_cols,
                plots_path,
                filter_zero_padding=not args.no_filter_zero_padding,
            )
        else:
            print('No numeric columns found; skipping plot export')

    if args.save_qpos:
        qpos_path = output_dir / 'qpos_joints_plots.png'
        save_qpos_plots(
            df, qpos_path, filter_zero_padding=not args.no_filter_zero_padding)

    if args.save_action:
        action_path = output_dir / 'action_joints_plots.png'
        save_action_plots(
            df,
            action_path,
            filter_zero_padding=not args.no_filter_zero_padding,
        )

    if (not args.save_video and not args.save_frames and not args.save_plots
            and not args.save_qpos and not args.save_action):
        print('\nNo save options specified; saving all outputs by default...')
        image_cols = get_image_columns(df)
        numeric_cols = get_numeric_columns(df)

        if image_cols:
            video_dir = output_dir / 'videos'
            save_images_video(df, image_cols, video_dir, fps=args.fps)

        if numeric_cols:
            plots_path = output_dir / 'numeric_data_plots.png'
            save_numeric_data_plots(
                df,
                numeric_cols,
                plots_path,
                filter_zero_padding=not args.no_filter_zero_padding,
            )

        qpos_path = output_dir / 'qpos_joints_plots.png'
        save_qpos_plots(
            df, qpos_path, filter_zero_padding=not args.no_filter_zero_padding)

        action_path = output_dir / 'action_joints_plots.png'
        save_action_plots(
            df,
            action_path,
            filter_zero_padding=not args.no_filter_zero_padding,
        )

        print(f'\nAll visualization outputs saved to: {output_dir}')


if __name__ == '__main__':
    main()
