"""Store for word variant IDs after disambiguation."""

from typing import List, Optional

from src.stores.base_store import BaseStore


class DisambiguatedStore(BaseStore[List[Optional[int]]]):
    """
    Stores word variant IDs (integers) after disambiguation.
    Format: {"sid": int, "variants": [42, null, 156, ...]}

    Variant IDs are integers corresponding to indices in Vocabulary.word_variants.
    """

    def _parse_data(self, data: dict) -> List[Optional[int]]:
        return data["variants"]

    def _serialize_data(self, sid: int, data: List[Optional[int]]) -> dict:
        return {"sid": sid, "variants": data}
