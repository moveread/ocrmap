from typing_extensions import Mapping, Iterable, Callable, TypedDict, Unpack, NotRequired
from collections import Counter
from dataclasses import dataclass
import ocr_map as om

class Params(TypedDict):
  alpha: NotRequired[int]
  k: NotRequired[int]
  edit_distance: NotRequired[Callable[[str, str], float] | None]

@dataclass
class LikelihoodMixin:

  Pocr: Mapping[str, Counter[str]]

  @property
  def labels(self) -> set[str]:
    """Set of labels in the training samples"""
    return set(self.Pocr.keys())
  
  def likelihood(self, label: str, **params: Unpack[Params]) -> Counter[str]:
    """Generalize the trained distribution to any `label`.
    - `alpha`: scaling factor for similarity (higher `alpha`s make the results closer to the original `label`)
    - `k`: number of similar words to consider
    """
    return om.generalize_distrib(label, self.Pocr, **params)

class Likelihood(LikelihoodMixin):
  """Distribution of OCR errors (from True Labels to OCR Predictions, aka the likelihood of labels given OCR predictions)"""
  @classmethod
  def fit(cls, samples: Iterable[om.Sample]) -> 'Likelihood':
    """Fit the OCR simulator to a set of samples"""
    return Likelihood(om.Pocr(samples))
  
@dataclass
class PosteriorMixin:
  Pl: Counter[str]
  Pocr_post: Mapping[str, Counter[str]]

  def posterior(self, ocrpred: str, **params: Unpack[Params]) -> Counter[str]:
    """Generalize the trained posterior distribution to any `ocrpred`"""
    return om.generalize_distrib(ocrpred, self.Pocr_post, **params)
  
  def denoise(self, ocrpreds: Iterable[tuple[str, float]], **params: Unpack[Params]) -> Counter[str]:
    """Denoise an entire OCR distribution"""
    out: dict[str, float] = Counter() # type: ignore
    for w, p in ocrpreds:
      for w2, p2 in self.posterior(w, **params).most_common(params.get('k', 25)):
        out[w2] += p * p2 
    return out # type: ignore
  
class Posterior(PosteriorMixin):
  """Posterior distribution of OCR errors (from OCR Predictions to True Labels, based on the prior distribution of labels)"""
  @classmethod
  def fit(cls, Pocr: Mapping[str, Counter[str]], labels: Iterable[str]) -> 'Posterior':
    """Fit the model given the likelihood and the label occurrences (frequencies used to compute their prior)"""
    Pl = om.Pl(labels)
    Pocr_post = om.Pocr_posterior(Pocr, Pl)
    return Posterior(Pl, Pocr_post)
  
@dataclass
class Model(LikelihoodMixin, PosteriorMixin):

  @staticmethod
  def fit(samples: Iterable[om.Sample]) -> 'Model':
    """Fit the model to a set of samples"""
    from itertools import tee
    s1, s2 = tee(samples)
    Pocr = om.Pocr(s1)
    Pl = om.Pl(l for l, _ in s2)
    Pocr_post = om.Pocr_posterior(Pocr, Pl)
    return Model(Pocr=Pocr, Pl=Pl, Pocr_post=Pocr_post)
  
  @staticmethod
  def unpickle(path: str) -> 'Model':
    """Load a model from a pickle file"""
    import pickle
    with open(path, 'rb') as f:
      model = pickle.load(f)
    assert isinstance(model, Model)
    return model
  
  def pickle(self, path: str) -> None:
    """Save the model to a pickle file"""
    import pickle
    with open(path, 'wb') as f:
      pickle.dump(self, f)


  def simulate(self, label: str, **params: Unpack[Params]) -> Counter[str]:
    """Simulate the OCR distribution, then denoise it. (simulates the result of denoising OCR preds)"""
    k = params.get('k') or 25
    likelihood = self.likelihood(label, **params)
    out = Counter()
    for w, p in likelihood.most_common(k):
      for w2, p2 in self.posterior(w, **params).most_common(k):
        out[w2] += p * p2
    return out