"""Store for vocab entry IDs after disambiguation."""

from typing import List, Optional

from src.stores.base_store import BaseStore


class DisambiguatedStore(BaseStore[List[Optional[int]]]):
    """
    Stores vocab entry IDs (integers) after disambiguation.
    Format: {"sid": int, "variants": [42, null, 156, ...]}

    Entry IDs are integers corresponding to indices in Vocabulary.vocab_entries.
    """

    def _parse_data(self, data: dict) -> List[Optional[int]]:
        return data["variants"]

    def _serialize_data(self, sid: int, data: List[Optional[int]]) -> dict:
        return {"sid": sid, "variants": data}
