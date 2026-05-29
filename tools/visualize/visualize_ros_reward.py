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
"""Real-time visualization of 4-camera views with multi-source reward overlay.

Supports any reward model that publishes ``RewardSignal`` on
``/robot/reward_signal`` (e.g. Qwen3-VL or future models).
Right-side panel switches between reward sources via **Tab** key.

Usage:
    source <DAGGER_REPO>/devel/setup.bash
    python3 tools/visualize/visualize_ros_reward.py \
        [--tile_width 320] [--tile_height 240] ...

Examples:
    python3 tools/visualize/visualize_ros_reward.py \
        --tile_width 320 --tile_height 240 \
        --display_fps 30
    python3 tools/visualize/visualize_ros_reward.py \
        --cameras h,l,r \
        --reward_sources qwen3_reward@online \
        --display_fps 30

Keyboard:
    q     - quit
    p     - save screenshot
    Tab   - switch reward source panel

The catkin environment must be sourced so ``dagger`` and ``dagger_msgs`` can
be imported normally.
"""

from __future__ import annotations
import argparse
import os
from abc import ABC, abstractmethod
from collections import deque
from contextlib import suppress
from datetime import datetime
from functools import partial
from typing import Dict, List, Optional

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image

from dagger.runtime_config import as_str, load_runtime_config
from dagger_msgs.msg import RewardSignal

_CONFIG = load_runtime_config()
DEFAULT_TOPICS = _CONFIG.camera_topics_by_short_key()
CAMERA_LABELS = _CONFIG.camera_labels_by_short_key()
CAMERA_ORDER = _CONFIG.camera_short_key_order()

BG_MAIN = (48, 46, 44)
BG_CARD = (58, 56, 54)
BG_TILE_EMPTY = (68, 66, 64)
CHART_PLOT_BG = (52, 50, 48)
TEXT_PRIMARY = (245, 243, 240)
TEXT_SECONDARY = (168, 165, 160)
TEXT_TERTIARY = (118, 115, 112)
ACCENT = (210, 155, 85)
ACCENT_MUTED = (120, 95, 60)
BORDER_SUBTLE = (78, 75, 72)
COLOR_GREEN = (105, 188, 108)
COLOR_RED = (105, 98, 220)
COLOR_YELLOW = (115, 205, 230)
COLOR_ORANGE = (95, 155, 245)
COLOR_DARK_RED = (85, 75, 190)
COLOR_GRAY = (145, 142, 138)
COLOR_CHART = ACCENT
COLOR_GRID = (62, 60, 58)
COLOR_SEP = BORDER_SUBTLE
COLOR_TAB_ACTIVE = (68, 66, 74)
COLOR_TAB_INACTIVE = (50, 48, 46)

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_TITLE = cv2.FONT_HERSHEY_DUPLEX

BG_DARK = BG_MAIN
TEXT_WHITE = TEXT_PRIMARY
TEXT_LIGHT = TEXT_SECONDARY
TEXT_DIM = TEXT_SECONDARY

QWEN3_STATUS_COLORS = {
    'Normal': COLOR_GREEN,
    'Static': COLOR_YELLOW,
    'Grasp fail': COLOR_RED,
    'Shaking': COLOR_ORANGE,
    'Drop': COLOR_DARK_RED,
}

QWEN3_STATUS_REWARD_MAP = {
    'Normal': 1.0,
    'Static': 0.3,
    'Grasp fail': 0.0,
    'Shaking': 0.1,
    'Drop': 0.0,
}

# OpenCV putText + Hershey fonts: stick to ASCII-only strings.
# This avoids glitches / stderr spam on some HighGUI backends.
CV_WINDOW_NAME = 'FluxDAgger Live Reward'


def _cv_safe_text(s: str) -> str:
    """Strip non-ASCII text before drawing with OpenCV Hershey fonts."""
    if not s:
        return ''
    return ''.join(c if ord(c) < 128 else '?' for c in s)


def ros_image_to_bgr(msg: Image) -> np.ndarray:
    h, w = msg.height, msg.width
    enc = msg.encoding.lower()
    data = np.frombuffer(msg.data, dtype=np.uint8)

    if enc == 'bgr8':
        frame = data.reshape(h, w, 3)
    elif enc == 'rgb8':
        frame = cv2.cvtColor(data.reshape(h, w, 3), cv2.COLOR_RGB2BGR)
    elif enc in ('mono8', '8uc1'):
        frame = cv2.cvtColor(data.reshape(h, w), cv2.COLOR_GRAY2BGR)
    elif enc == 'bgra8':
        frame = cv2.cvtColor(data.reshape(h, w, 4), cv2.COLOR_BGRA2BGR)
    elif enc == 'rgba8':
        frame = cv2.cvtColor(data.reshape(h, w, 4), cv2.COLOR_RGBA2BGR)
    else:
        raise ValueError(f'Unsupported encoding: {msg.encoding}')

    return np.ascontiguousarray(frame)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def infer_qwen3_status(value: float) -> str:
    """Infer Qwen3-VL status label from reward value."""
    for status, rv in QWEN3_STATUS_REWARD_MAP.items():
        if abs(value - rv) < 0.01:
            return status
    return 'Unknown'


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=('Realtime multi-camera + multi-source reward '
                     'visualization.'))
    parser.add_argument('--tile_width', type=int, default=320)
    parser.add_argument('--tile_height', type=int, default=240)
    parser.add_argument('--panel_width', type=int, default=400)
    parser.add_argument('--display_fps', type=float, default=15.0)
    parser.add_argument('--history_len', type=int, default=100)
    parser.add_argument(
        '--cameras',
        type=str,
        default='h,l,r',
        help=('Comma-separated camera keys to show. Allowed: f,h,l,r. '
              'Default is h,l,r (the standard 3-camera setup). '
              'Add f for front view, e.g. --cameras f,h,l,r.'),
    )
    parser.add_argument(
        '--reward_topic',
        type=str,
        default='/robot/reward_signal',
    )
    parser.add_argument(
        '--reward_sources',
        type=str,
        default='',
        help=('Comma-separated reward source tags to show. '
              'Empty means auto-discover all sources. '
              'Example: --reward_sources qwen3_reward@online'),
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=as_str(_CONFIG.paths.screenshots_dir),
    )
    parser.add_argument('--queue_size', type=int, default=5)
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Less ROS console output (log level WARN).',
    )
    return parser


def render_camera_tile(frame, label: str, tile_w: int,
                       tile_h: int) -> np.ndarray:
    if frame is None:
        tile = np.full((tile_h, tile_w, 3), BG_TILE_EMPTY, dtype=np.uint8)
        text = 'No Signal'
        tw = cv2.getTextSize(text, FONT, 0.55, 1)[0][0]
        cv2.putText(
            tile,
            text,
            ((tile_w - tw) // 2, tile_h // 2),
            FONT,
            0.55,
            TEXT_TERTIARY,
            1,
            cv2.LINE_AA,
        )
    else:
        tile = cv2.resize(
            frame, (tile_w, tile_h), interpolation=cv2.INTER_LINEAR)

    bar_h = 30
    overlay = tile[:bar_h, :].copy()
    cv2.rectangle(tile, (0, 0), (tile_w, bar_h), BG_CARD, -1)
    cv2.addWeighted(overlay, 0.22, tile[:bar_h, :], 0.78, 0, tile[:bar_h, :])
    cv2.line(tile, (0, bar_h - 1), (tile_w, bar_h - 1), ACCENT_MUTED, 1,
             cv2.LINE_AA)
    cv2.putText(tile, label, (10, 21), FONT_TITLE, 0.48, TEXT_PRIMARY, 1,
                cv2.LINE_AA)

    return tile


TAB_BAR_H = 36


def render_tab_bar(keys: List[str], active_idx: int, width: int) -> np.ndarray:
    """Render horizontal tab bar for switching reward sources."""
    bar = np.full((TAB_BAR_H, width, 3), BG_CARD, dtype=np.uint8)
    cv2.line(bar, (0, TAB_BAR_H - 1), (width, TAB_BAR_H - 1), BORDER_SUBTLE, 1,
             cv2.LINE_AA)
    if not keys:
        cv2.putText(bar, 'Reward sources - waiting', (12, 24), FONT, 0.45,
                    TEXT_TERTIARY, 1, cv2.LINE_AA)
        return bar

    tab_w = min(width // len(keys), 200)
    x = 0
    for i, key in enumerate(keys):
        is_active = (i == active_idx)
        bg = COLOR_TAB_ACTIVE if is_active else COLOR_TAB_INACTIVE
        cv2.rectangle(bar, (x, 0), (x + tab_w, TAB_BAR_H - 1), bg, -1)
        cv2.line(bar, (x + tab_w, 0), (x + tab_w, TAB_BAR_H - 1),
                 BORDER_SUBTLE, 1, cv2.LINE_AA)

        label = _cv_safe_text(key[:20] if len(key) > 20 else key)
        color = TEXT_PRIMARY if is_active else TEXT_SECONDARY
        tw = cv2.getTextSize(label, FONT, 0.42, 1)[0][0]
        cv2.putText(bar, label, (x + (tab_w - tw) // 2, 23), FONT, 0.42, color,
                    1, cv2.LINE_AA)

        if is_active:
            cv2.rectangle(bar, (x, TAB_BAR_H - 3), (x + tab_w, TAB_BAR_H),
                          ACCENT, -1)
        x += tab_w

    return bar


class RewardPanel(ABC):
    """Base class for a single reward source visualization panel."""

    def __init__(self, source: str, width: int, height: int, history_len: int):
        self.source = source
        self.width = width
        self.height = height
        self.reward_history: deque = deque(maxlen=history_len)
        self.latest_reward = RewardSignal()

    def update(self, msg: RewardSignal) -> None:
        self.latest_reward = msg
        self.reward_history.append(msg.value)

    @abstractmethod
    def render(self) -> np.ndarray:
        ...

    def _render_background(self) -> np.ndarray:
        return np.full((self.height, self.width, 3), BG_DARK, dtype=np.uint8)

    def _render_title(self, panel: np.ndarray, y: int, pad: int,
                      title: str) -> int:
        cv2.putText(panel, _cv_safe_text(title), (pad, y + 28), FONT_TITLE,
                    0.58, TEXT_PRIMARY, 1, cv2.LINE_AA)
        y += 32
        rule_w = min(96, self.width - 2 * pad)
        cv2.rectangle(panel, (pad, y), (pad + rule_w, y + 2), ACCENT, -1)
        y += 8
        cv2.line(panel, (pad, y), (self.width - pad, y), BORDER_SUBTLE, 1,
                 cv2.LINE_AA)
        return y + 2

    def _render_done(self, panel: np.ndarray, y: int, pad: int,
                     done: bool) -> int:
        y += 10
        cv2.putText(panel, 'Episode status', (pad, y + 12), FONT, 0.4,
                    TEXT_TERTIARY, 1, cv2.LINE_AA)
        y += 22
        color = COLOR_GREEN if done else COLOR_RED
        label = 'Terminal (done)' if done else 'In progress'

        cx, cy = pad + 9, y + 9
        cv2.circle(panel, (cx, cy), 7, color, -1, cv2.LINE_AA)
        cv2.circle(panel, (cx, cy), 7, BORDER_SUBTLE, 1, cv2.LINE_AA)
        cv2.putText(panel, label, (cx + 16, cy + 5), FONT, 0.5, TEXT_PRIMARY,
                    1, cv2.LINE_AA)

        y = cy + 20
        cv2.line(panel, (pad, y), (self.width - pad, y), BORDER_SUBTLE, 1,
                 cv2.LINE_AA)
        return y + 1

    def _render_chart(self, panel: np.ndarray, y_top: int, y_bottom: int,
                      pad: int) -> None:
        y_top += 10
        cv2.putText(panel, 'Reward trajectory', (pad, y_top + 12), FONT, 0.4,
                    TEXT_TERTIARY, 1, cv2.LINE_AA)

        chart_t = y_top + 22
        chart_b = y_bottom - 8
        chart_l = pad + 38
        chart_r = self.width - pad
        ch = chart_b - chart_t
        cw = chart_r - chart_l

        if ch < 20 or cw < 20:
            return

        cv2.rectangle(panel, (chart_l, chart_t), (chart_r, chart_b),
                      CHART_PLOT_BG, -1)
        cv2.rectangle(panel, (chart_l, chart_t), (chart_r, chart_b),
                      BORDER_SUBTLE, 1)

        for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
            gy = chart_b - int(tick * ch)
            cv2.line(panel, (chart_l, gy), (chart_r, gy), COLOR_GRID, 1,
                     cv2.LINE_AA)
            lbl = f'{int(tick * 100)}%'
            cv2.putText(panel, lbl, (pad, gy + 4), FONT, 0.32, TEXT_TERTIARY,
                        1, cv2.LINE_AA)

        data = list(self.reward_history)
        if len(data) < 2:
            return

        n = len(data)
        pts = []
        for i, v in enumerate(data):
            x = chart_l + int(i / max(n - 1, 1) * cw)
            y = chart_b - int(max(0.0, min(1.0, v)) * ch)
            pts.append((x, y))

        for i in range(len(pts) - 1):
            cv2.line(panel, pts[i], pts[i + 1], ACCENT_MUTED, 3, cv2.LINE_AA)
        for i in range(len(pts) - 1):
            cv2.line(panel, pts[i], pts[i + 1], COLOR_CHART, 2, cv2.LINE_AA)

        cv2.circle(panel, pts[-1], 5, TEXT_PRIMARY, -1, cv2.LINE_AA)
        cv2.circle(panel, pts[-1], 5, ACCENT, 1, cv2.LINE_AA)

    def _render_metadata(self, panel: np.ndarray, y_top: int, pad: int,
                         episode_id: int, step_id: int,
                         stamp_sec: float) -> None:
        y = y_top + 4
        cv2.line(panel, (pad, y), (self.width - pad, y), BORDER_SUBTLE, 1,
                 cv2.LINE_AA)
        y += 14

        cv2.putText(panel, f'Episode {episode_id}  Step {step_id}',
                    (pad, y + 10), FONT, 0.38, TEXT_SECONDARY, 1, cv2.LINE_AA)
        y += 18

        if stamp_sec > 0:
            ts = datetime.fromtimestamp(stamp_sec).strftime('%H:%M:%S.%f')[:-3]
        else:
            ts = '--:--:--.---'
        cv2.putText(panel, ts, (pad, y + 10), FONT, 0.38, TEXT_TERTIARY, 1,
                    cv2.LINE_AA)

    @staticmethod
    def create(source: str, width: int, height: int,
               history_len: int) -> 'RewardPanel':
        """Factory: select panel subclass based on source tag."""
        src_lower = source.lower()
        if 'qwen3' in src_lower:
            return Qwen3RewardPanel(source, width, height, history_len)
        else:
            return GenericRewardPanel(source, width, height, history_len)


class Qwen3RewardPanel(RewardPanel):
    """Panel for Qwen3-VL reward status and history."""

    STATUS_HISTORY_LEN = 30

    def __init__(self, source: str, width: int, height: int, history_len: int):
        super().__init__(source, width, height, history_len)
        self._status_history: deque = deque(maxlen=self.STATUS_HISTORY_LEN)

    def update(self, msg: RewardSignal) -> None:
        super().update(msg)
        status = infer_qwen3_status(msg.value)
        self._status_history.append(status)

    def render(self) -> np.ndarray:
        panel = self._render_background()
        rw = self.latest_reward
        pad = 20
        y = 0

        title = self._format_title()
        y = self._render_title(panel, y, pad, title)
        y = self._render_done(panel, y, pad, rw.done)
        y = self._render_status_section(panel, y, pad)

        chart_bottom = self.height - 50
        self._render_chart(panel, y, chart_bottom, pad)
        self._render_metadata(panel, chart_bottom, pad, rw.episode_id,
                              rw.step_id, rw.stamp.to_sec())
        return panel

    def _format_title(self) -> str:
        # e.g. "qwen3_reward@online_node" -> "Qwen3 reward"
        return 'Qwen3 reward'

    def _render_status_section(self, panel: np.ndarray, y: int,
                               pad: int) -> int:
        y += 10
        cv2.putText(panel, 'Semantic status', (pad, y + 12), FONT, 0.4,
                    TEXT_TERTIARY, 1, cv2.LINE_AA)
        y += 24

        current = (
            self._status_history[-1] if self._status_history else 'Waiting...')
        if current in QWEN3_STATUS_COLORS:
            color = QWEN3_STATUS_COLORS[current]
        else:
            color = COLOR_GRAY

        cv2.putText(panel, _cv_safe_text(current), (pad, y + 22), FONT_TITLE,
                    0.72, color, 1, cv2.LINE_AA)
        y += 36

        if self._status_history:
            bar_x0 = pad
            bar_x1 = self.width - pad
            bar_h = 16
            n = len(self._status_history)
            slot_w = max(2, (bar_x1 - bar_x0) // max(n, 1))

            cv2.rectangle(panel, (bar_x0, y), (bar_x1, y + bar_h),
                          CHART_PLOT_BG, -1)
            x = bar_x0
            for i, status in enumerate(self._status_history):
                c = QWEN3_STATUS_COLORS.get(status, COLOR_GRAY)
                x_end = min(x + slot_w, bar_x1)
                cv2.rectangle(panel, (x, y), (x_end, y + bar_h), c, -1)
                x = x_end

            cv2.rectangle(panel, (bar_x0, y), (bar_x1, y + bar_h),
                          BORDER_SUBTLE, 1)
            y += bar_h + 4

            legend_y = y
            lx = pad
            for status_name in ('Normal', 'Static', 'Grasp fail', 'Shaking',
                                'Drop'):
                c = QWEN3_STATUS_COLORS[status_name]
                cv2.rectangle(panel, (lx, legend_y), (lx + 8, legend_y + 8), c,
                              -1)
                cv2.rectangle(panel, (lx, legend_y), (lx + 8, legend_y + 8),
                              BORDER_SUBTLE, 1)
                short = status_name.split()[0][:4]
                cv2.putText(panel, short, (lx + 12, legend_y + 9), FONT, 0.3,
                            TEXT_TERTIARY, 1, cv2.LINE_AA)
                lx += 12 + cv2.getTextSize(short, FONT, 0.3, 1)[0][0] + 8

            y = legend_y + 16
        else:
            cv2.putText(panel, 'Waiting for inference...', (pad, y + 8), FONT,
                        0.45, TEXT_TERTIARY, 1, cv2.LINE_AA)
            y += 20

        y += 6
        cv2.line(panel, (pad, y), (self.width - pad, y), BORDER_SUBTLE, 1,
                 cv2.LINE_AA)
        return y + 1


class GenericRewardPanel(RewardPanel):
    """Minimal generic panel for any reward source.

    Shows the title, done status, and reward history chart.
    """

    def render(self) -> np.ndarray:
        panel = self._render_background()
        rw = self.latest_reward
        pad = 20
        y = 0

        tag = self.source.split('@')[0] if '@' in self.source else self.source
        title = _cv_safe_text(f'{tag} Reward')
        y = self._render_title(panel, y, pad, title)
        y = self._render_done(panel, y, pad, rw.done)

        chart_bottom = self.height - 50
        self._render_chart(panel, y, chart_bottom, pad)
        self._render_metadata(panel, chart_bottom, pad, rw.episode_id,
                              rw.step_id, rw.stamp.to_sec())
        return panel


class RosRewardVisualizer:

    def __init__(self, args):
        self.args = args
        self.tile_w = args.tile_width
        self.tile_h = args.tile_height

        self.active_keys = self._resolve_active_keys(args)
        rospy.loginfo('Active cameras: %s', self.active_keys)

        self.latest_frames = {k: None for k in self.active_keys}

        self.reward_panels: Dict[str, RewardPanel] = {}
        self.panel_keys: List[str] = []
        self.active_tab_idx = 0
        self._allowed_sources: Optional[set] = None
        if args.reward_sources:
            self._allowed_sources = {
                s.strip()
                for s in args.reward_sources.split(',') if s.strip()
            }

        n = len(self.active_keys)
        grid_cols = 2 if n > 1 else 1
        grid_rows = (n + grid_cols - 1) // grid_cols
        self.panel_h = self.tile_h * grid_rows - TAB_BAR_H
        self.grid_cols = grid_cols
        self.grid_rows = grid_rows

        for key in self.active_keys:
            topic = DEFAULT_TOPICS[key]
            rospy.Subscriber(
                topic,
                Image,
                self._make_cam_cb(key),
                queue_size=args.queue_size,
            )
            rospy.loginfo('[%s] Subscribed: %s', key, topic)

        rospy.Subscriber(
            args.reward_topic,
            RewardSignal,
            self._reward_cb,
            queue_size=args.queue_size,
        )
        rospy.loginfo('Subscribed reward: %s', args.reward_topic)

        ensure_dir(args.output_dir)
        self.screenshot_count = 0

    @staticmethod
    def _resolve_active_keys(args):
        """Parse --cameras; empty string triggers auto-detect."""
        if args.cameras:
            keys = [k.strip() for k in args.cameras.split(',') if k.strip()]
            for k in keys:
                if k not in DEFAULT_TOPICS:
                    allowed = list(DEFAULT_TOPICS.keys())
                    raise ValueError(
                        f'Invalid camera key: {k}. Allowed: {allowed}')
            return keys

        rospy.loginfo(
            'Auto-detecting active cameras (waiting up to 3s per topic)...')
        active = []
        for key in CAMERA_ORDER:
            topic = DEFAULT_TOPICS[key]
            try:
                rospy.wait_for_message(topic, Image, timeout=3.0)
                active.append(key)
                rospy.loginfo('  [%s] %s - OK', key, topic)
            except rospy.ROSException:
                rospy.logwarn('  [%s] %s - not publishing, skipped', key,
                              topic)
        if not active:
            raise RuntimeError(
                'No camera topics detected. Check your sensor drivers.')
        return active

    def _make_cam_cb(self, key):
        return partial(self._camera_cb, key)

    def _camera_cb(self, key, msg):
        self.latest_frames[key] = ros_image_to_bgr(msg)

    def _reward_cb(self, msg: RewardSignal) -> None:
        source = msg.source
        if not source:
            return

        if self._allowed_sources and source not in self._allowed_sources:
            return

        if source not in self.reward_panels:
            panel = RewardPanel.create(source, self.args.panel_width,
                                       self.panel_h, self.args.history_len)
            self.reward_panels[source] = panel
            self.panel_keys.append(source)
            rospy.loginfo('New reward source detected: %s -> %s panel', source,
                          type(panel).__name__)

        self.reward_panels[source].update(msg)

    def _get_active_panel(self) -> Optional[RewardPanel]:
        if not self.panel_keys:
            return None
        idx = self.active_tab_idx % len(self.panel_keys)
        return self.reward_panels[self.panel_keys[idx]]

    def render(self) -> np.ndarray:
        tiles = [
            render_camera_tile(self.latest_frames[k], CAMERA_LABELS[k],
                               self.tile_w, self.tile_h)
            for k in self.active_keys
        ]
        while len(tiles) < self.grid_rows * self.grid_cols:
            tiles.append(
                np.full((self.tile_h, self.tile_w, 3), BG_DARK,
                        dtype=np.uint8))

        rows = []
        for r in range(self.grid_rows):
            rows.append(
                np.hstack(tiles[r * self.grid_cols:(r + 1) * self.grid_cols]))
        cam_grid = np.vstack(rows)

        grid_w = self.tile_w * self.grid_cols
        for r in range(1, self.grid_rows):
            cv2.line(cam_grid, (0, self.tile_h * r), (grid_w, self.tile_h * r),
                     BORDER_SUBTLE, 1, cv2.LINE_AA)
        for c in range(1, self.grid_cols):
            cv2.line(cam_grid, (self.tile_w * c, 0),
                     (self.tile_w * c, cam_grid.shape[0]), BORDER_SUBTLE, 1,
                     cv2.LINE_AA)

        right_w = self.args.panel_width
        total_h = cam_grid.shape[0]

        tab_bar = render_tab_bar(
            self.panel_keys,
            self.active_tab_idx % max(len(self.panel_keys), 1), right_w)

        panel = self._get_active_panel()
        if panel is not None:
            panel_img = panel.render()
        else:
            panel_img = np.full((total_h - TAB_BAR_H, right_w, 3),
                                BG_MAIN,
                                dtype=np.uint8)
            msg = 'Awaiting /robot/reward_signal'
            tw = cv2.getTextSize(msg, FONT, 0.48, 1)[0][0]
            cy = panel_img.shape[0] // 2
            cv2.putText(panel_img, msg, ((right_w - tw) // 2, cy - 6),
                        FONT_TITLE, 0.48, TEXT_SECONDARY, 1, cv2.LINE_AA)
            sub = 'Multi-source panels appear when messages arrive.'
            tw2 = cv2.getTextSize(sub, FONT, 0.36, 1)[0][0]
            cv2.putText(panel_img, sub, ((right_w - tw2) // 2, cy + 18), FONT,
                        0.36, TEXT_TERTIARY, 1, cv2.LINE_AA)

        right_panel = np.vstack([tab_bar, panel_img])

        if right_panel.shape[0] != total_h:
            right_panel = cv2.resize(right_panel, (right_w, total_h))
        cv2.rectangle(right_panel, (0, 0), (right_w - 1, total_h - 1),
                      BORDER_SUBTLE, 1)

        sep = np.full((total_h, 2, 3), BG_CARD, dtype=np.uint8)
        sep[:, 1:2, :] = ACCENT_MUTED

        canvas = np.hstack((cam_grid, sep, right_panel))
        gh, gw = cam_grid.shape[:2]
        cv2.rectangle(canvas, (0, 0), (gw - 1, gh - 1), BORDER_SUBTLE, 1)
        return canvas

    def save_screenshot(self, canvas: np.ndarray):
        ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        path = os.path.join(self.args.output_dir, f'screenshot_{ts}.png')
        cv2.imwrite(path, canvas, [cv2.IMWRITE_PNG_COMPRESSION, 2])
        self.screenshot_count += 1
        rospy.loginfo('Screenshot saved: %s (total=%d)', path,
                      self.screenshot_count)

    def run(self):
        rate = rospy.Rate(self.args.display_fps)
        rospy.loginfo(
            'Visualization running at %.1f FPS. '
            "Press 'q' to quit, 'p' for screenshot, Tab to switch reward.",
            self.args.display_fps,
        )

        # Use one named window to avoid duplicate HighGUI windows.
        cv2.namedWindow(CV_WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

        while not rospy.is_shutdown():
            canvas = self.render()
            cv2.imshow(CV_WINDOW_NAME, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                rospy.loginfo('Quit requested.')
                break
            elif key == ord('p'):
                self.save_screenshot(canvas)
            elif key == 9:  # Tab key
                if self.panel_keys:
                    self.active_tab_idx = ((self.active_tab_idx + 1) %
                                           len(self.panel_keys))
                    rospy.loginfo('Switched to reward source: %s',
                                  self.panel_keys[self.active_tab_idx])
            rate.sleep()

        with suppress(Exception):
            cv2.destroyWindow(CV_WINDOW_NAME)
        rospy.loginfo('Visualization stopped. Screenshots=%d',
                      self.screenshot_count)


def main():
    args, _ = build_parser().parse_known_args()
    log_level = rospy.WARN if args.quiet else rospy.INFO
    rospy.init_node(
        'ros_reward_visualizer',
        anonymous=True,
        log_level=log_level,
    )
    viz = RosRewardVisualizer(args)
    viz.run()


if __name__ == '__main__':
    main()
