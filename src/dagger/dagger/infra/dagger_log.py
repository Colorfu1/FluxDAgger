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
"""Small ROS logging facade with consistent DAgger message prefixes."""

from __future__ import annotations
from typing import Any, Dict, Optional, Tuple

import rospy

__all__ = [
    'DaggerLogger',
    'bold_green',
    'bold_yellow',
    'get_logger',
]

_logger_cache: Dict[Tuple[str, str], 'DaggerLogger'] = {}


def _default_prefix(name: str) -> str:
    if not name:
        return 'dagger'
    parts = name.rsplit('.', 1)
    return parts[-1] if parts else name


def _body(msg: str, args: Tuple[Any, ...]) -> str:
    if args:
        return msg % args
    return msg


def bold_yellow(text: str) -> str:
    """Return bold yellow ANSI text."""
    return f'\033[1;33m{text}\033[0m'


def bold_green(text: str) -> str:
    """Return bold green ANSI text."""
    return f'\033[1;32m{text}\033[0m'


def get_logger(name: str = __name__,
               *,
               prefix: Optional[str] = None) -> 'DaggerLogger':
    """Return a logger with a stable message prefix.

    Args:
        name: Logger name, usually ``__name__``.
        prefix: Prefix shown before log messages. Defaults to the last segment
            of ``name``.
    """
    pref = prefix if prefix is not None else _default_prefix(name)
    key = (name, pref)
    if key not in _logger_cache:
        _logger_cache[key] = DaggerLogger(name, pref)
    return _logger_cache[key]


class DaggerLogger:
    """Logger that routes messages to ROS rosout."""

    def __init__(self, _name: str, prefix: str) -> None:
        self._prefix = prefix

    def _full(self, msg: str, args: Tuple[Any, ...]) -> str:
        return f'[{self._prefix}] {_body(msg, args)}'

    def debug(self, msg: str, *args: Any) -> None:
        rospy.logdebug(self._full(msg, args))

    def info(self, msg: str, *args: Any) -> None:
        rospy.loginfo(self._full(msg, args))

    def info_green(self, msg: str, *args: Any) -> None:
        """Log an info message with a green highlighted body."""
        self.info('%s', bold_green(_body(msg, tuple(args))))

    def info_yellow(self, msg: str, *args: Any) -> None:
        """Log an info message with a yellow highlighted body."""
        self.info('%s', bold_yellow(_body(msg, tuple(args))))

    def warning(self, msg: str, *args: Any) -> None:
        rospy.logwarn(self._full(msg, args))

    def error(self, msg: str, *args: Any) -> None:
        rospy.logerr(self._full(msg, args))

    def debug_throttle(self, period: float, msg: str, *args: Any) -> None:
        rospy.logdebug_throttle(period, '%s', self._full(msg, args))

    def info_throttle(self, period: float, msg: str, *args: Any) -> None:
        rospy.loginfo_throttle(period, '%s', self._full(msg, args))

    def warning_throttle(self, period: float, msg: str, *args: Any) -> None:
        rospy.logwarn_throttle(period, '%s', self._full(msg, args))

    def error_throttle(self, period: float, msg: str, *args: Any) -> None:
        rospy.logerr_throttle(period, '%s', self._full(msg, args))
