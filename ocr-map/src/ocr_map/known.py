from collections import defaultdict, Counter
from typing import Iterable, Mapping
from .types import Sample

def Pocr(samples: Iterable[Sample]) -> dict[str, Counter[str]]:
  """`Pocr(p | l)` = probability of `p` being recognized as label `l`.
  - Returns a dictionary such that `dict[lab][pred] = Pocr(lab | pred)`
  """
  Pocr = defaultdict(Counter)

  for label, top_preds in samples:
    for word, prob in top_preds:
      Pocr[label][word] += prob # type: ignore
  
  # normalize distributions
  for label, counter in Pocr.items():
    total = sum(counter.values())
    for k in counter:
      counter[k] /= total # type: ignore

  return Pocr

def Pl(labels: Iterable[str]) -> Counter[str]:
  """`Pl(l)` = prior probability of label `l` (just relative frequencies)"""
  freqs = Counter(labels)
  total = sum(freqs.values())
  for label in freqs:
    freqs[label] /= total # type: ignore

  return freqs

def Pocr_posterior(Pocr: Mapping[str, Counter[str]], Pl: dict[str, float] | Counter[str]) -> dict[str, Counter[str]]:
  """`Pocr(l | p)` = probability of label `l` given that `p` was recognized, given `Pocr` and the prior `Pl` over all labels
  - Returns a dictionary such that `dict[pred][lab] = Pocr(lab | pred)`
  """
  Pocr_post = defaultdict(Counter)

  for lab, preds in Pocr.items():
    for pred, prob in preds.items():
      Pocr_post[pred][lab] = prob * Pl[lab] # type: ignore

  # normalize
  for pred, counter in Pocr_post.items():
    total = sum(counter.values())
    if total == 0:
      continue # overflowed to 0, so let's just skip it
    for label in counter:
      Pocr_post[pred][label] /= total # type: ignore

  return Pocr_post
