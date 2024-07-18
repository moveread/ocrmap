from typing import NamedTuple, Sequence

TopPreds = Sequence[tuple[str, float]]
"""Top predictions `(word, prob)`, sorted by decreasing probability."""

class Sample(NamedTuple):
  label: str
  top_preds: TopPreds
