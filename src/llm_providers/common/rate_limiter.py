"""Async sliding-window rate limiter for API calls."""

import asyncio
import time
from collections import deque


class AsyncRateLimiter:
    """Limits requests to max_per_minute using a sliding window."""

    def __init__(self, max_per_minute: int):
        self.max_per_minute = max_per_minute
        self.timestamps: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self):
        """Wait until a request slot is available."""
        if self.max_per_minute <= 0:
            return

        async with self._lock:
            now = time.monotonic()

            # Remove timestamps older than 60 seconds
            while self.timestamps and now - self.timestamps[0] > 60:
                self.timestamps.popleft()

            # Wait if at capacity
            if len(self.timestamps) >= self.max_per_minute:
                sleep_time = 60 - (now - self.timestamps[0])
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                    # Clean up again after sleeping
                    now = time.monotonic()
                    while self.timestamps and now - self.timestamps[0] > 60:
                        self.timestamps.popleft()

            self.timestamps.append(time.monotonic())
