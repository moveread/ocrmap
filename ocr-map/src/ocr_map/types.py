from typing import TypeAlias, Sequence

TopPreds: TypeAlias = Sequence[tuple[str, float]]
"""Top predictions `(word, prob)`, sorted by decreasing probability."""
Sample: TypeAlias = tuple[str, TopPreds]
"""`(label, top_preds)`"""
