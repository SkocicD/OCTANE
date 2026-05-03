"""NetworkGuard — heartbeat-based connection watchdog.

The comm node calls heartbeat_ok() on every successful heartbeat send and
heartbeat_fail() on every exception.  Once consecutive failures reach the
threshold, _dispatch() fires all registered handlers exactly once per
disconnect event.  Handlers are plain callables — no ROS dependency here.

Usage in network_comm_node:

    self._guard = NetworkGuard(fail_threshold=3)
    self._guard.register(self._kill_video_feed)

    # in _send_heartbeat success path:
    self._guard.heartbeat_ok()

    # in _send_heartbeat except path:
    self._guard.heartbeat_fail()
"""

import logging

log = logging.getLogger(__name__)


class NetworkGuard:

    def __init__(self, fail_threshold: int = 3):
        self._fail_threshold = fail_threshold
        self._fail_count     = 0
        self._triggered      = False
        self._handlers: list = []

    # ── registration ──────────────────────────────────────────────────────────

    def register(self, fn) -> None:
        """Register a zero-argument callable to call on connection loss."""
        self._handlers.append(fn)

    # ── heartbeat feed ────────────────────────────────────────────────────────

    def heartbeat_ok(self) -> None:
        self._fail_count = 0
        self._triggered  = False

    def heartbeat_fail(self) -> None:
        self._fail_count += 1
        if self._fail_count >= self._fail_threshold and not self._triggered:
            self._triggered = True
            self._dispatch()

    # ── dispatch ──────────────────────────────────────────────────────────────

    def _dispatch(self) -> None:
        log.warning('NetworkGuard: %d consecutive heartbeat failures — running handlers', self._fail_count)
        for fn in self._handlers:
            try:
                fn()
            except Exception as exc:
                log.error('NetworkGuard handler %s raised: %s', fn, exc)
