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
"""Small worker-thread helper for asynchronous inference.

``InferenceWorker`` keeps ROS callbacks lightweight by moving expensive work
to one background thread. It is intentionally small: one input queue, one
inference function, and an optional result callback.
"""

from __future__ import annotations
import queue
import threading
from contextlib import suppress
from typing import Any, Callable

from dagger.infra.dagger_log import get_logger

InferenceFn = Callable[[Any], Any]
ResultHandler = Callable[[Any, Any], None]

__all__ = ['InferenceWorker']


class InferenceWorker:
    """Run latest-item inference in a dedicated daemon thread.

    Queue semantics are optimized for online perception: ``submit`` never
    blocks, and when the queue is full the oldest queued item is dropped so the
    worker eventually sees the freshest observation.
    """

    _SENTINEL = object()
    _POLL_TIMEOUT_SEC = 0.5

    def __init__(
        self,
        infer_fn: InferenceFn,
        *,
        name: str = 'inference_worker',
        queue_size: int = 1,
    ) -> None:
        if queue_size < 1:
            raise ValueError('queue_size must be >= 1')

        self._infer_fn = infer_fn
        self._name = name
        self._log = get_logger(__name__, prefix=name)

        self._queue: queue.Queue[Any] = queue.Queue(maxsize=queue_size)
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._result_handler: ResultHandler | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_result_handler(self, handler: ResultHandler) -> None:
        """Register ``handler(input_obj, result)`` for non-``None`` results."""
        self._result_handler = handler

    def start(self) -> None:
        """Start the worker thread. Calling this repeatedly is safe."""
        if self._is_running:
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name=self._name,
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        """Ask the worker to stop and wait briefly for it to exit."""
        self._stop_event.set()
        self._put_latest(self._SENTINEL)

        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def submit(self, obj: Any) -> bool:
        """Queue work without blocking.

        Returns:
            ``True`` if an older queued item was dropped, otherwise ``False``.
        """
        if self._stop_event.is_set():
            return False
        return self._put_latest(obj)

    # ------------------------------------------------------------------
    # Queue handling
    # ------------------------------------------------------------------

    @property
    def _is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _put_latest(self, obj: Any) -> bool:
        """Put ``obj`` into the queue, dropping one stale item if needed."""
        try:
            self._queue.put_nowait(obj)
            return False
        except queue.Full:
            with suppress(queue.Empty):
                self._queue.get_nowait()

        # The consumer can race with this branch. Keep the operation
        # best-effort and non-blocking, matching submit's public contract.
        with suppress(queue.Full):
            self._queue.put_nowait(obj)
        return True

    # ------------------------------------------------------------------
    # Worker loop
    # ------------------------------------------------------------------

    def _run_loop(self) -> None:
        while True:
            try:
                item = self._queue.get(timeout=self._POLL_TIMEOUT_SEC)
            except queue.Empty:
                continue

            if item is self._SENTINEL:
                break

            self._process_one(item)

    def _process_one(self, item: Any) -> None:
        """Run inference and dispatch one non-empty result."""
        try:
            result = self._infer_fn(item)
        except Exception as exc:  # noqa: BLE001 - keep worker alive
            self._log.warning('Inference failed: %s', exc)
            return

        if result is None or self._result_handler is None:
            return

        try:
            self._result_handler(item, result)
        except Exception as exc:  # noqa: BLE001 - keep worker alive
            self._log.warning('Result handler failed: %s', exc)
