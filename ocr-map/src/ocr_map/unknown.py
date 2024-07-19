from typing import Iterable, Mapping, Callable
from collections import Counter
import math
from editdistance import eval as ed

def sim(a: str, b: str, *, alpha: float = 10, ed = ed):
  ned = ed(a, b) / max(len(a), len(b), 1) # just in case
  return math.exp(-alpha * ned)

def Psim(p: str, Vp: Iterable[str], *, alpha: float = 10, ed = ed) -> Counter[str]:
  """Computes `Psim^p (w)` over all `w in Vp`
  - `p`: possibly out-of-vocabulary word to test
  - `Vp`: vocabulary to test against
  """
  Psim = Counter[str]({ w: sim(p, w, alpha=alpha, ed=ed) for w in Vp })
  total = sum(Psim.values())
  for w in Psim:
    Psim[w] /= total # type: ignore

  return Psim

def generalize_distrib(
  w: str, P: Mapping[str, Counter[str]], *,
  alpha: float = 10, k: int = 25,
  edit_distance: Callable[[str, str], float] | None = None
):
  """Generalizes `P(w)` given `P`, by finding similar words to `w` in the vocabulary of `P`
  - `w`: word to generalize
  - `P`: existing distribution to generalize
  - `alpha`: scaling factor for similarity
  - `k`: number of similar words to consider
  - `edit_distance`: custom edit distance function (defaults to the good-old Levenshtein distance)
  """
  Vp = list(P.keys())
  Psim_w = Psim(w, Vp, alpha=alpha, ed=edit_distance or ed)
  posterior = Counter[str]()

  for p, Psim_pw in Psim_w.most_common(k):
    for l, Pocr_lp in P[p].items():
      posterior[l] += Psim_pw * Pocr_lp

  total = sum(posterior.values())
  for l in posterior:
    posterior[l] /= total # type: ignore

  return posterior