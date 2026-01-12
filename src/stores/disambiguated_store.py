"""Store for word variant IDs after disambiguation."""

from typing import List, Optional

from src.stores.base_store import BaseStore


class DisambiguatedStore(BaseStore[List[Optional[str]]]):
    """
    Stores word variant IDs after disambiguation.
    Format: {"sid": int, "variants": ["000123_0", null, "000456_1", ...]}
    """

    def _parse_data(self, data: dict) -> List[Optional[str]]:
        return data["variants"]

    def _serialize_data(self, sid: int, data: List[Optional[str]]) -> dict:
        return {"sid": sid, "variants": data}
