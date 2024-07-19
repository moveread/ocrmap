# OCR ↔ Vocabulary

# Usage

**CLI**
```bash
cat ocrpreds.ndjson | ocrmap fit model.pkl
cat labels.txt | ocrmap simulate --model model.pkl > fakepreds.ndjson
cat ocrpreds.ndjson | ocrmap denoise -m model.pkl > label-distribs.ndjson
```

**Python**
```python
from ocrmap import Model

model = Model.fit(samples)
# or
model = Model.unpickle('model.pkl')

model.posterior('ee4') # Counter({ 'ed4': 0.7, 'e4': 0.1, ... })
model.denoise({ 'ee4': 0.7, 'e4': 0.1, ... }) # Counter({ 'ed4': 0.65, 'e4': 0.15, ... })
```

# 0. Abstract

The Chess Transformer has a word-level vocabulary, consisting of all legal SANs plus some common styles and languages. However, the OCR output vocabulary is character-level. Here, we tackle some of the problems thus created:

1. How to map the OCR vocabulary to the transformer's
2. How to simulate common OCR errors (to train the transformer to be robust to them)

# 1. Introduction

## 1.1. High level approach

We explore an empiric-first approach. We take our foundational dataset (of just over 100k image-label samples) and obtain the top-25 OCR predictions for each of them.

We'll use these to both generate fake OCR noise and map real OCR outputs to words in the transformer's vocabulary. Whenever some word or OCR prediction is not in our samples, we'll use an edit distance with custom substitution costs to compute the closest match.

## 1.2. Notations

- *Chess Transformer Vocabulary* $V_T$: all legal SANs, with a few common styles and translations
- *OCR Output Vocabulary* $V_O$: all possible sequences of characters, numbers, and the hyphen ('-')
- *Known OCR Predictions Vocabulary* $V_P$: set of OCR predictions generated over our OCR dataset
- *Known OCR Labels Vocabulary* $V_L$: set of words from $V_T$ that also appear in our OCR dataset (and thus have some OCR predictions associated) ($V_L\subseteq V_T$)

# 2. Estimating Probabilities

## 2.1. Known OCR Predictions

We iterate our dataset and obtain the top $k=25$ predictions for each sample. For each of the $N$ samples, we obtain the label $l_i$ and the top predictions $(p_i^1,\pi_i^1),...,(p_i^k,\pi_i^k)$ with the respective probabilities ($\pi_i^j$) assigned by the OCR model.

Now, for each label $l\in V_L$, we'll estimate a distribution $P_\text{ocr}(p∣l)$ over $V_P$. This estimates “how likely is the OCR model to estimate $p$ given that $l$ is the true label”.

We'll use each row $i\in[N]$ as part of the distribution for $l_i$. Concretely, we define $P^i_\text{ocr}(p_i^j∣l_i)=\pi_i^j$ and $P^i_\text{ocr}(p∣l_i)=0$ for all other $p\in V_P$.

Then, we define $P_\text{ocr}(p|l)$ as the sum of all the $P^i_\text{ocr}$ such that $l=l_i$. We divide by a normalizing factor $S$ to ensure all probabilities add up to 1.

$$
P_\text{ocr}(p∣l)=\frac{1}{S}\sum_{i:l_i=l}P^i_\text{ocr}(p∣l_i)
$$

Now, we're actually interested in obtaining $P_\text{ocr}(l|p)$, so that we can estimate $l$'s given $p$'s. Using Bayes' Theorem:

$$
P_\text{ocr}(l∣p)=\frac{P_\text{ocr}(p∣l)P(l)}{P(p)}
$$

$P(l)$ will be empirically computed on our dataset (and is quite a relevant prior since labels have quite unbalanced frequencies).

Using $P(l)$, $P(p)$ can be computed by marginalizing over all labels $l\in V_L$.

$$
P(p)=\sum_{l\in V_L}P_\text{ocr}(p∣l)P(l)
$$

### Dataset Statistics

Let us compare our relatively limited dataset (100k moves from about 1.4k games) with 1M games from Lichess:

| Lichess | Original Dataset |
| ------------------------ | --------------------------------- |
| ![Untitled](media/lichess-freqs.png) | ![Untitled](media/mr-freqs.png) |

Surprisingly (to me), short castle (O-O) is the most frequent move! And by quite a margin; about 2% of the moves are O-O.

| Lichess | Original Dataset |
| ------------------------ | --------------------------------- |
| ![Untitled](media/lichess-loglog.png) | ![Untitled](media/mr-loglog.png) |

Zipfian? Meh... Anyways, it seems our distribution represents quite well the real distribution (or the distribution of random lichess games, anyways).

### Empirical Analysis

How good are these predictions? Let's just take a few random words in $V_P$, and show the top 5 words in their distributions:

```
h=7 -> a1=D (0.90), fxg5 (0.08), h7 (0.03)
Ngg7 -> Ng7 (1.00)
Q8d7 -> Qxd7 (1.00)
Chxf3 -> Chxf5 (0.75), Chf3 (0.25)
Ncc1 -> Nc4 (0.79), Nc1 (0.21)
Bea5 -> Be5 (0.55), Bxa6 (0.19), Bxa5 (0.18), Bxe5 (0.08)
bac8 -> Rbc8 (0.62), b3 (0.27), Tac8 (0.11)
RgB -> BxN (0.48), Rg8 (0.40), Rg1 (0.07), Rg3 (0.03), Rg4 (0.01)
Agd3 -> Ad3 (0.50), Ag3 (0.35), Axd3 (0.15)
Dxa1 -> Dxa1 (0.52), Axa1 (0.12), Dxc1 (0.09), Dxb1 (0.03), Dxe6 (0.03)
```

Not bad!

### Discussion

With $P_\text{ocr}(l∣p)$, we can map OCR predictions $p\in V_P$ to the input vocabulary $V_T$. Similarly, with $P_\text{ocr}(p∣l)$ we can generate OCR-like noise in $V_L$.

However, $V_P\subset  V_O$ (and will always be, since $V_O$ is infinite, and we can only have a finite number of words in our dataset $V_P$) and $V_L \subseteq V_T$ (though the gap may significantly close as we collect more data).

Thus, we need an alternative way to define $P_\text{ocr}$ over $V_O$ and $V_T$.

## 2.2. Unknown OCR Predictions

To estimate $P_\text{ocr}$ over all $V_O$ and $V_T$ we use a custom-cost normalized edit distance to map $V_O\rightarrow V_P$ and $V_T\rightarrow V_L$. Given the edit distance $d$ and a scaling parameter $\alpha$, we define the probability measure $P_\text{sim}^{(p)}(w)$ over all $V_O$, that estimates “how likely are the predictions of $w$ to be close to those of $p$”.

First, we define a similarity measure:

$$
\mathrm{sim}(w,p)=\exp(-\alpha d(w, p))
$$

Note that $\mathrm{sim}(w,w)=e^0=1$, whilst the limit as $d(w,p)\rightarrow\infty$ is $e^{-\infty}=0$. We normalize this function over all $p$'s to obtain the probability measure, $P_\text{sim}^{(p)}$. Given a normalizing factor $S$:

$$
P_\text{sim}^{(p)}=\frac{1}{S}\mathrm{sim}(w,p)=\frac{1}{S}\exp(-\alpha d(w, p))
$$

Now we're in a position to estimate the general $P_\text{ocr}(l∣w)$ for all $w\in V_O$. We simply compute a weighted average of the original $P_\text{ocr}$ respect to the similarities $P_\text{sim}^{(p)}$:

$$
P_\text{ocr}(l∣w)=\sum_{p\in V_P}P_\text{sim}^{(p)}(w)\cdot P_\text{ocr}(l∣p)
$$

Since all $P_\text{sim}$'s are already normalized, the resulting distribution will be so as well (??? Seems true in practice, but I can't quite believe than on paper)

Similarly, we can generalize $P_\text{ocr}(p∣t)$ for all $t\in V_T$, so that we can generate OCR-like noise for any word in the transformer's vocabulary.

$$
P_\text{ocr}(p∣t)=\sum_{l\in V_L}P_\text{sim}^{(l)}(t)\cdot P_\text{ocr}(p∣l)
$$