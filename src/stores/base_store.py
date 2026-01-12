"""Base class for index-aligned sentence stores."""

import gzip
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generator, Generic, List, Optional, Tuple, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class BaseStoreWriter:
    """Context manager for writing to a store."""

    def __init__(self, path: Path, serialize_fn):
        self.path = path
        self._serialize_fn = serialize_fn
        self._file = None
        self._count = 0

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = gzip.open(self.path, "wt", encoding="utf-8")
        self._count = 0
        return self

    def write(self, sid: int, data):
        """Write a single sentence's data."""
        json_data = self._serialize_fn(sid, data)
        self._file.write(json.dumps(json_data, ensure_ascii=False) + "\n")
        self._count += 1
        if self._count % 100000 == 0:
            logger.info(f"Written {self._count:,} entries to {self.path.name}")

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._file:
            self._file.close()
            logger.info(f"Finalized {self.path.name} with {self._count:,} entries")
        return False


class BaseStore(ABC, Generic[T]):
    """Abstract base class for index-aligned sentence stores."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._count: Optional[int] = None

    def exists(self) -> bool:
        """Check if store file exists."""
        return self.path.exists()

    def count(self) -> int:
        """Get total number of sentences (scans file on first call, then caches)."""
        if self._count is not None:
            return self._count
        if not self.exists():
            return 0
        count = 0
        with gzip.open(self.path, "rt", encoding="utf-8") as f:
            for _ in f:
                count += 1
        self._count = count
        return count

    def __iter__(self) -> Generator[Tuple[int, T], None, None]:
        """Iterate over (sid, data) pairs."""
        if not self.exists():
            raise FileNotFoundError(f"Store not found: {self.path}")
        with gzip.open(self.path, "rt", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                yield data["sid"], self._parse_data(data)

    def iter_batched(
        self, batch_size: int
    ) -> Generator[List[Tuple[int, T]], None, None]:
        """Iterate in batches of (sid, data) tuples."""
        batch = []
        for item in self:
            batch.append(item)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def open_writer(self) -> BaseStoreWriter:
        """Open a writer context manager."""
        return BaseStoreWriter(self.path, self._serialize_data)

    @abstractmethod
    def _parse_data(self, data: dict) -> T:
        """Parse JSON data into the store's data type."""
        pass

    @abstractmethod
    def _serialize_data(self, sid: int, data: T) -> dict:
        """Serialize data for JSON storage."""
        pass
