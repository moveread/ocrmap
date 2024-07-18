from typing import Iterable, TextIO
from . import Sample

def parse_ndjson(samples: TextIO) -> Iterable[Sample]:
  """Parses samples from a file with JSON lines formatted as `{"label": "label", "preds": top_preds}`"""
  import orjson
  for line in samples:
    obj = orjson.loads(line)
    yield Sample(obj['label'], obj['preds'])
