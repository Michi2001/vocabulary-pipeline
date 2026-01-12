"""Store for original sentence tokens."""

from typing import List

from src.stores.base_store import BaseStore


class SentenceStore(BaseStore[List[str]]):
    """
    Stores original sentence tokens.
    Format: {"sid": int, "tokens": ["El", "gato", ...]}
    """

    def _parse_data(self, data: dict) -> List[str]:
        return data["tokens"]

    def _serialize_data(self, sid: int, data: List[str]) -> dict:
        return {"sid": sid, "tokens": data}
