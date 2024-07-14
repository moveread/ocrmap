from typing import Mapping, Iterable, Sequence, Callable
from collections import Counter
from dataclasses import dataclass
import ocr_map as om

def sample(distribution: Counter):
  import random
  return random.choices(list(distribution.keys()), weights=distribution.values())[0] # type: ignore

@dataclass
class LikelihoodMixin:

  Pocr: Mapping[str, Counter[str]]

  @property
  def labels(self) -> set[str]:
    """Set of labels in the training samples"""
    return set(self.Pocr.keys())
  
  def likelihood(
    self, label: str, *, alpha: int = 10, k: int = 25,
    edit_distance: Callable[[str, str], float] | None = None  
  ) -> Counter[str]:
    """Generalize the trained distribution to any `label`"""
    return om.generalize_distrib(label, self.Pocr, alpha=alpha, k=k, edit_distance=edit_distance)

  def simulate(
    self, word: str, *, alpha: int = 10, k: int = 25,
    edit_distance: Callable[[str, str], float] | None = None
  ) -> str:
    """Simulate OCR noise for any given word
    - `alpha`: scaling factor for similarity (higher `alpha`s make the results closer to the original `word`)
    - `k`: number of similar words to consider
    """
    return sample(self.likelihood(word, alpha=alpha, k=k, edit_distance=edit_distance))


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

  def posterior(
    self, ocrpred: str, *, alpha: int = 10, k: int = 25,
    edit_distance: Callable[[str, str], float] | None = None
  ) -> Counter[str]:
    """Generalize the trained posterior distribution to any `ocrpred`"""
    return om.generalize_distrib(ocrpred, self.Pocr_post, alpha=alpha, k=k, edit_distance=edit_distance)
  
  def denoise(
    self, ocrpred: str, *, alpha: int = 10, k: int = 25,
    edit_distance: Callable[[str, str], float] | None = None
  ) -> str:
    """Denoise any OCR prediction `ocrpred`"""
    return self.posterior(ocrpred, alpha=alpha, k=k, edit_distance=edit_distance).most_common(1)[0][0]
  
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
  def fit(samples: Sequence[om.Sample]) -> 'Model':
    """Fit the model to a set of samples"""
    Pocr = om.Pocr(samples)
    Pl = om.Pl(l for l, _ in samples)
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