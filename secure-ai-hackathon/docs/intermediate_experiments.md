# Intermediate track — experiment log

Working notes for the Intermediate writeup: what we tried, what worked, what didn't, and why.
Raw numbers for every run are in `experiments/results/intermediate.jsonl`.

## Setup and rules

- **Task:** 5 "banks" train the notebook's `WeakMLP` (41 → 8 → 1) with federated learning on a
  non-IID split of NSL-KDD. The split is Dirichlet α=0.3 by attack family, with partition seed 42
  as in the notebook. Evaluation is F1 on KDDTest+ at a 0.5 threshold, after 8 rounds of 1 local
  epoch (SGD, lr 0.05, batch 256).
- **Score:** F1 of our aggregation minus F1 of naive (size-weighted) FedAvg on the same non-IID split.
- **Privacy rule we hold ourselves to:** the server sees only what the notebook harness already
  gives it — each bank's trained model and its dataset size. **No label or attack-family
  statistics leave a bank.**
- **Seeds:** the data split is fixed, and each configuration is run with training seeds 0–4. We
  report mean ± std of final-round F1. `torch.manual_seed` is set right before each run, so results
  don't depend on which cells ran earlier.
- **Code:** `experiments/harness.py` is a verbatim copy of the notebook's setup cells.
  `experiments/common.py` adds a multi-seed runner and an extended FL loop. That loop reproduces
  the notebook's `run_fl` exactly when its extra options are off (checked: identical F1 on all 5
  seeds).

## Diagnosis: why naive FedAvg loses F1 under skew

Rows per bank, broken down by traffic family:

| Bank | Rows | normal | dos | probe | r2l/u2r/other | FedAvg weight |
|---|---|---|---|---|---|---|
| 0 | 6,272 | 1,177 | 1,213 | **3,752** | 130 | 5.7% |
| 1 | 10,658 | 1,443 | 2,964 | **5,747** | **504** | 9.6% |
| 2 | 48,219 | 15,203 | **32,673** | 343 | 0 | 43.5% |
| 3 | 23,239 | **19,855** | 3,250 | 79 | 55 | 20.9% |
| 4 | 22,585 | **21,653** | 382 | 336 | 214 | 20.4% |

- **Knowledge and voting weight don't match.** Banks 0 and 1 hold ~90% of all probe traffic and
  ~70% of the rare R2L/U2R attacks, but size-weighted FedAvg gives them only ~15% of the vote.
  Banks 3 and 4 are almost entirely normal traffic and get ~41%.
- **Client drift.** One local epoch means ~190 SGD steps for bank 2 but only ~25 for bank 0. The
  big banks drift much further toward their own local optimum ("mostly normal" or "DoS vs
  normal") before averaging.
- **The test set is harder than training.** KDDTest+ is 57% attacks (training pool: 46%), and
  13% of it is R2L/U2R/novel attacks (training: under 1%). Recall is the bottleneck even in the
  IID setting.
- **What the skew costs:** almost all of it is recall (0.63 → 0.57). Precision stays at ~0.95.

## Stage 1 — baselines and simple re-weighting (`experiments/01_baselines.py`)

| Configuration | F1 (mean ± std, 5 seeds) | Precision | Recall |
|---|---|---|---|
| IID, naive FedAvg (reference) | 0.7556 ± 0.0064 | 0.937 | 0.633 |
| **non-IID, naive FedAvg (baseline)** | **0.7147 ± 0.0086** | 0.951 | 0.573 |
| non-IID, equal weights | 0.7491 ± 0.0128 | 0.929 | 0.628 |
| non-IID, log(size) weights | 0.7443 ± 0.0135 | 0.932 | 0.620 |
| non-IID, sqrt(size) weights | 0.7254 ± 0.0082 | 0.941 | 0.590 |

**What worked:** flattening the size weighting. Equal weights recover +3.4 F1, almost the whole
IID gap, and use *less* information than FedAvg (sizes aren't needed at all). The ordering size <
sqrt < log < equal matches the diagnosis: the less the big normal-heavy banks dominate, the
better the recall on attacks.

**Caveat:** equal weighting only works because, in this split, the small banks happen to be
the attack-diverse ones. It's a blunt fix, and a tiny bank with junk data would get the same
say as a large one. That motivates weights that depend on the *updates* themselves, which we
try in stage 2.

## Exploratory results we are *not* using (privacy)

Early exploration also tried weights computed from each bank's label/family histograms. These
are recorded here only as reference points, because they break our privacy rule:

| Configuration (needs histograms — excluded) | F1 | Why excluded / why it matters |
|---|---|---|
| Notebook's hint: size × closeness of bank's attack rate to global rate | 0.7137 ± 0.0106 | Needs label counts; it also **didn't help** — the binary attack/normal balance isn't the problem, *which kinds* of attack a bank has seen is |
| Family-coverage weights (each attack family gets an equal share of the vote) | 0.7580 ± 0.0122 | Needs per-family counts from every bank. It shows what knowing the data would buy: a private method close to ~0.758 is doing about as well as an oracle weighting |

Also tried: server momentum (FedAvgM) on size weights, β=0.5 → 0.7171, β=0.9 with server
lr 0.5 → 0.7319. Small gains at most: momentum smooths noise from round to round, but our loss
comes from *systematically* wrong weights, not noise.

## Stage 2 — privacy-safe improvements (`experiments/02_privacy_safe.py`)

Baseline for comparison: naive FedAvg on non-IID gets **0.7147 ± 0.0086**.

### 2a. Fixing client drift

| Configuration | Where it runs | F1 | Precision | Recall |
|---|---|---|---|---|
| FedNova, size weights | server | 0.7476 ± 0.0159 | 0.930 | 0.625 |
| **FedNova, equal weights** | **server** | **0.7787 ± 0.0082** | 0.918 | 0.676 |
| FedProx μ=0.01, size weights | client | 0.7149 ± 0.0086 | 0.950 | 0.573 |
| FedProx μ=0.1, size weights | client | 0.7186 ± 0.0062 | 0.947 | 0.579 |
| FedProx μ=1.0, size weights | client | 0.7463 ± 0.0158 | 0.960 | 0.611 |
| FedProx μ=0.1, equal weights | client | 0.7541 ± 0.0117 | 0.927 | 0.636 |
| Cap of 25 local steps/round, size weights | client | 0.7532 ± 0.0145 | 0.944 | 0.627 |
| **Cap of 25 local steps/round, equal weights** | client | **0.7857 ± 0.0138** | 0.934 | 0.678 |

**FedNova** (Wang et al., 2020) divides each bank's update by its number of local SGD steps
(τᵢ = ⌈nᵢ / 256⌉, which the server can compute from the size), averages these per-step
directions, and rescales by the average τ. It runs entirely on the server and needs nothing
beyond sizes.

**What worked — drift is half the story.** With size weights, bank 2 takes ~190 steps per round
and bank 0 only ~25. So bank 2's update is both *weighted* 7× more and *~7× longer*: its
influence is counted twice. FedNova removes the second factor (+3.3 F1 alone). Combined with
equal weights, which removes the first, it reaches **0.779, above the IID reference (0.756)**
and above the privacy-violating family-coverage oracle (0.758).

Capping local steps attacks the same double-counting from the client side and does slightly
better (0.786), but it changes the client protocol. Each round, big banks train on only ~6.4k
random rows instead of all of theirs. FedNova is the cleaner "fix the aggregation" answer.

**What didn't work: FedProx.** It is the standard answer to client drift, but it barely moves
F1 until μ=1.0, and even then it is worse than FedNova. The proximal term slows *every* bank
equally. The problem isn't that banks drift; it's that the big banks drift *more* and then
also get counted more. FedProx doesn't address that imbalance.

### 2b. Server-side adaptive optimizers (FedAdam / FedYogi)

| Configuration | F1 |
|---|---|
| FedAdam, server lr 0.01, size / equal weights | 0.7481 / 0.7716 |
| FedAdam, server lr 0.03, size / equal weights | 0.7313 / 0.7604 |
| FedAdam, server lr 0.1, size / equal weights | 0.7161 / 0.7388 |
| FedYogi | within 0.0003 of FedAdam in every setting |

A modest gain, but only at a small server learning rate, and seed variance is high (±0.019).
With only 8 rounds, Adam's moment estimates barely warm up. Yogi and Adam are indistinguishable
in so few steps. Not worth the extra hyperparameters.

### 2c. Weights based on the updates themselves

| Configuration | F1 |
|---|---|
| Normalized directions (unit-norm deltas, mean-norm step) | 0.7540 ± 0.0093 |
| Normalized directions (median-norm step) | 0.7477 ± 0.0148 |
| Dissimilarity weights, temperature 1 | 0.7539 ± 0.0131 |
| **Dissimilarity weights, temperature 3** | **0.7676 ± 0.0099** |

**Dissimilarity weighting**: weight = exp(−T · mean cosine similarity of this bank's update
to the others'). Banks whose update disagrees with the majority carry knowledge the majority
lacks, so they get up-weighted. Without seeing any data, it rediscovers what the histograms
told us: the attack-diverse banks (0, 1) push in a different direction from the normal-heavy ones.

**Security tension (important for the Advanced track):** this is the *opposite* of what
robust aggregation does. Krum or cosine filters down-weight the odd client out, because
that's what an attacker looks like. Under skew, the odd client out is often the most valuable
honest one. Any Advanced-track defense has to deal with this.

## Stage 3 — combining the winners, checking they generalise (`experiments/03_combine_and_robustness.py`)

### 3a. Combinations and temperature (notebook split, 5 seeds)

| Configuration | F1 | Precision | Recall |
|---|---|---|---|
| Dissimilarity T=5 | 0.7815 ± 0.0166 | 0.919 | 0.680 |
| Dissimilarity T=10 | 0.7383 ± **0.0682** | 0.947 | 0.618 |
| FedNova + dissimilarity T=1 | 0.7841 ± 0.0085 | 0.915 | 0.686 |
| FedNova + dissimilarity T=3 | 0.7967 ± 0.0094 | 0.908 | 0.710 |
| FedNova + dissimilarity T=5 | 0.8051 ± 0.0090 | 0.891 | 0.734 |

Taken at face value, FedNova + dissimilarity T=5 is the best result (+9.0 F1). Stage 4 shows
why we **don't** pick it. Note also that dissimilarity alone becomes unstable at T=10
(std 0.068): the weights collapse onto a single bank.

### 3b. Does it hurt when data is IID?

Every candidate gives 0.7554–0.7556 on the IID split, the same as naive FedAvg. That's
expected: with IID banks of equal size, equal weights = size weights, all τᵢ are equal (so
FedNova = FedAvg), and all updates are near-identical (so dissimilarity weights stay near
uniform). **The methods only change anything when there is skew.**

### 3c. Longer training (20 rounds instead of 8)

| Configuration | 8 rounds | 20 rounds |
|---|---|---|
| Naive FedAvg | 0.7147 | 0.7259 |
| Equal weights | 0.7491 | 0.7572 |
| FedNova (equal) | 0.7787 | 0.7764 |
| Dissimilarity T=3 | 0.7676 | 0.7670 |
| FedNova + dissimilarity T=3 | 0.7967 | 0.7914 |

The gap isn't an artifact of stopping early: naive FedAvg gains only ~1 point with 2.5× the
rounds, and the gains of the improved methods persist.

### 3d. Other non-IID splits: is the notebook's split special?

The obvious objection to equal weights is that they only work because *this* split puts the
diverse data in the small banks. So we re-ran every candidate on 9 fresh splits: Dirichlet
α ∈ {0.1 (extreme), 0.3 (notebook), 1.0 (mild)} × partition seeds {1, 2, 3}, with training seeds
0–2. Raw results are in `results/intermediate_splits.jsonl`; `summarize_splits.py` prints the tables.

**Mean F1 gain over naive FedAvg on the same split:**

| Method | α=0.1 | α=0.3 | α=1.0 | Beats naive FedAvg | Worst split |
|---|---|---|---|---|---|
| **FedNova (equal)** | **+0.090** | **+0.052** | +0.032 | **9/9** | **+0.013** |
| Equal weights | +0.050 | +0.037 | +0.023 | 9/9 | +0.008 |
| FedNova + dissim T=3 | +0.040 | +0.022 | +0.034 | 6/9 | −0.058 |
| FedNova + dissim T=5 | −0.085 | −0.032 | +0.031 | 5/9 | −0.249 |
| Dissimilarity T=3 | −0.078 | −0.010 | +0.008 | 6/9 | −0.295 |

**Findings**

- **FedNova (equal) is the robust winner.** It improves on naive FedAvg on every split, and the
  more skewed the data, the bigger the gain (+9 F1 at α=0.1). That's what you'd expect from a
  method that fixes the size/step double-counting: stronger skew makes the double-counting worse.
- **Plain equal weights are also always positive, but smaller.** FedNova adds +1.5 to +4 points on
  top of them at every skew level.
- **Dissimilarity weighting is unreliable.** It helps on some splits and collapses on others: on
  α=0.1 split 3 it drops to 0.43. On α=0.1 split 2, T=5 ends with recall 1.000 / precision 0.616,
  i.e. it predicts *everything* as an attack. That's the same collapse onto one bank as in stage 4,
  here landing on a bank that is ~100% attacks. Its best-case numbers on the notebook split were luck.

**Bug found along the way: empty banks.** On α=0.1 split 2 one bank receives **zero rows**. Its
"update" is all zeros, so FedNova divided by τ=0 and the cosine weighting divided by a zero
norm. The resulting NaN model scored F1 = 0 for every method except plain averaging. We fixed it
with `active_only()` in `common.py`, which skips banks that contributed nothing this round
before aggregating. That's also realistic hygiene: in a real federation a bank can drop out or
send an empty update. The guard does nothing on splits without empty banks, so every other
result is unaffected. The table above uses the re-run with the guard.

## Stage 4 — is the gain real, or a threshold shift? (`experiments/04_diagnostics.py`)

F1 at a fixed 0.5 threshold on an attack-heavy test set (57% attacks) rewards any model that
simply leans toward predicting "attack". So we also measured metrics that don't depend on the
threshold, and checked the notebook's reserved **in-distribution holdout** (15k KDDTrain+ rows
that no bank ever trained on). All values are averaged over 5 seeds.

| Method | Test AUC | Best-threshold test F1 | Holdout AUC | Holdout F1 | False-positive rate | DoS recall | Probe recall | R2L/U2R recall |
|---|---|---|---|---|---|---|---|---|
| Naive FedAvg | 0.877 | 0.857 | 0.991 | 0.956 | 3.9% | 0.775 | 0.611 | 0.031 |
| Equal weights | 0.909 | 0.872 | 0.991 | 0.958 | 6.3% | 0.776 | 0.802 | 0.111 |
| **FedNova (equal)** | **0.920** | **0.879** | 0.990 | 0.948 | 7.9% | 0.778 | **0.910** | 0.227 |
| FedNova + dissim T=3 | 0.919 | 0.872 | 0.989 | 0.934 | 9.5% | 0.783 | 0.963 | 0.317 |
| FedNova + dissim T=5 | 0.916 | 0.868 | 0.986 | 0.920 | 11.8% | 0.793 | 0.985 | 0.380 |

**Findings**

1. **FedNova (equal) is the genuinely best model.** It has the highest test AUC and best-threshold
   F1, with in-distribution quality almost unchanged. Its gain is exactly where the diagnosis
   predicted: **probe recall 0.61 → 0.91** and **rare-attack recall 0.03 → 0.23**. That's the
   knowledge held by the small banks 0 and 1, which naive FedAvg drowned out. DoS recall, which
   every bank can learn, doesn't change.
2. **Adding dissimilarity weighting raises F1@0.5 but lowers AUC.** The extra F1 comes from
   predicting "attack" more often (false-positive rate 4% → 12%, holdout F1 0.956 → 0.920). That's
   a threshold shift that happens to suit this test set, not better learning.
3. **Dissimilarity weighting collapses onto one bank.** The per-round weights (seed 0, T=5):

   | Round | Bank 0 | Bank 1 | Bank 2 | Bank 3 | Bank 4 |
   |---|---|---|---|---|---|
   | 1 | 0.28 | 0.26 | 0.08 | 0.12 | 0.26 |
   | 4 | 0.18 | 0.56 | 0.07 | 0.09 | 0.10 |
   | 8 | 0.04 | **0.88** | 0.03 | 0.02 | 0.03 |
   | (FedAvg, for comparison) | 0.06 | 0.10 | 0.44 | 0.21 | 0.20 |

   It's a feedback loop: as the global model moves toward bank 1, the other banks' corrections
   all look alike, so bank 1 looks ever more "unique". By round 8 the federation is mostly
   training on one bank's data (86% attacks), which is why it over-predicts attacks.
4. **Security consequence (carry over to the Advanced track).** Bank 1 is the malicious bank in
   the Advanced scenario. An aggregator that rewards disagreement would hand an attacker almost
   the whole vote: a poisoned update is by construction the most "different" one. Under skew,
   *robustness* (down-weight outliers) and *fairness to minority knowledge* (up-weight
   outliers) pull in opposite directions.

### 3e. How much should size count? (FedNova with weights ∝ sizeᵞ)

| γ | 0 (equal) | 0.25 | 0.5 | 0.75 | 1 (size) |
|---|---|---|---|---|---|
| F1 | **0.7787** | 0.7699 | 0.7632 | 0.7562 | 0.7476 |

F1 falls steadily as size counts more. There's no hidden optimum in between, so equal weighting is
the best choice here, not an arbitrary endpoint.

## Current conclusion (Intermediate track)

**Chosen method: FedNova with equal client weights**, plus the `active_only` guard for empty banks.
It runs entirely on the server and uses nothing beyond what naive FedAvg already receives (the
models and sizes).

| | Naive FedAvg | FedNova (equal) | Gain |
|---|---|---|---|
| F1, notebook split (5 seeds) | 0.7147 ± 0.0086 | 0.7787 ± 0.0082 | **+6.4** |
| Test AUC | 0.877 | 0.920 | +4.3 |
| Probe recall | 0.61 | 0.91 | +30 |
| R2L/U2R recall | 0.03 | 0.23 | +20 |
| False-positive rate | 3.9% | 7.9% | +4.0 (the cost) |
| 9 other splits | — | beats FedAvg 9/9 | +3 to +9 depending on skew |

**Why it works, in one sentence:** under skew, size-weighted FedAvg counts large banks twice —
once through a larger weight and once through a longer update (more local SGD steps). So
the average is dominated by the big, normal-heavy banks, and the small banks' probe and
rare-attack knowledge is drowned out. FedNova removes the step-count effect, and equal weights
remove the size effect.

**Tried and rejected, with reasons:**

- The notebook's label-balance weighting: needs label counts (privacy), and doesn't help anyway.
- Family-coverage weights: excellent (0.758), but need per-family counts (privacy). Kept only as an oracle reference.
- FedProx: slows all banks equally, so it doesn't fix the imbalance.
- FedAdam and FedYogi: small gains with high variance; 8 rounds is too few for adaptive moments to warm up.
- Dissimilarity weighting: best raw F1 on the notebook split, but it collapses onto one bank, only shifts the threshold (lower AUC), is unstable across splits, and would hand an attacker the vote.

**Open caveats, to be honest about in the writeup:**

- FedNova trusts each bank's *reported* size to compute τᵢ. A bank that under-reports its size
  gets its update scaled *up* (the per-step direction is multiplied by the average τ). Naive FedAvg
  also trusts sizes, but in the opposite direction. Worth testing in the Advanced track.
- Part of the recall gain comes with a higher false-positive rate (4% → 8%). For a bank's intrusion
  detection that's usually a good trade (a missed intrusion costs more than an extra alert), but
  it is a trade.

## Submission notebook and headline numbers

The notebook is `02_Day2_Intermediate_Advanced.ipynb`: the organisers' notebook plus our cells, run top
to bottom with nbconvert and no errors. The exported model is in `submission_intermediate/`, and the
chart is `figures/intermediate_f1_over_rounds.png` (the cover image candidate).

**Headline (the writeup's main number):** the notebook split, training seed 42, both runs seeded
identically right before `run_fl`:

| | Naive FedAvg | FedNova (equal) | Improvement |
|---|---|---|---|
| **F1** | **0.7117** | **0.7821** | **+0.0704** |
| Precision | 0.9524 | 0.9173 | |
| Recall | 0.5681 | 0.6816 | |
| AUC | 0.8697 | 0.9102 | |
| Probe recall | 0.582 | 0.967 | |
| R2L/U2R recall | 0.025 | 0.187 | |
| False-positive rate | 3.8% | 8.1% | |

**Supporting numbers:** over 5 seeds, 0.7147 ± 0.0086 → 0.7787 ± 0.0082 (+0.064), and FedNova wins on every
seed (+0.053 to +0.070).

**Verified:** `submission_intermediate/model_scripted.pt`, loaded in a fresh process with
`torch.jit.load` and scored on the notebook-preprocessed KDDTest+, gives P 0.9173 / R 0.6816 /
F1 0.7821, identical to `submission.json`.

**Seed 42 was chosen in advance:** it's the notebook's own `SEED`. It wasn't picked from a sweep,
and our tuning used seeds 0–4.

**Observation from the chart:** FedNova's F1 is highest after round 1 (0.808, 5-seed mean) and eases
down to 0.779 by round 8, staying above both baselines throughout. Later rounds trade a little recall
for precision as the banks' updates start pulling against each other. We kept the notebook's 8 rounds
rather than tuning the number of rounds on the test set, since that would be tuning on test data.

## Stage 5 — can anything beat FedNova (equal)? (`experiments/05_round2.py`)

Every variant is still privacy-safe (models + sizes only). From here on we also log **test AUC**, so a
gain that only comes from shifting the threshold (the stage-4 trap) is visible. **Adoption rule:** a
variant has to beat FedNova (equal) on the notebook split *and* across the 9 alternative splits. We've
now tried many variants on one test set, so a single-split win is likely to be luck.

Notebook split, 5 seeds. Reference: naive FedAvg 0.7147 (AUC 0.877); **FedNova (equal) 0.7787 ± 0.0082 (AUC 0.920)**.

| Idea | Variant | F1 | AUC | Verdict |
|---|---|---|---|---|
| Server step length | median τ instead of mean τ | 0.7785 | 0.921 | Same (median ≈ mean here) |
| | 0.5 × mean τ | 0.7876 ± 0.0136 | 0.923 | Wins here, but **−6.9 on one of the 9 splits** — rejected |
| | 0.75 × mean τ | 0.7821 ± 0.0103 | 0.921 | Beats FedNova on **8/9 splits**, but only by +0.3 on average |
| | 1.5 × mean τ | 0.7712 | 0.922 | Worse |
| | 2.0 × mean τ | 0.7364 ± 0.0290 | 0.923 | Unstable (overshoots) |
| Server momentum | FedNova + β=0.3 / β=0.6 | 0.7721 / 0.7756 | 0.920 / 0.916 | No gain |
| Fairness weights (q-FFL proxy) | weights ∝ ‖per-step update‖^q, q=1/2/4 | 0.7654 / 0.7562 / 0.7588 | ~0.922 | Worse on F1 |
| Resolving conflicting updates (PCGrad/FedFV) | FedNova + projection | 0.7725 | 0.921 | No gain |
| | + ‖u‖² weights | 0.7727 | 0.925 | No gain on F1 |
| SCAFFOLD (client-side change) | size weights | 0.7738 ± 0.0045 | **0.927** | Strong, but not better on F1 |
| | equal weights | 0.7714 | 0.923 | |
| | + FedNova normalisation | 0.7680 | 0.923 | |

**Findings**

- **Plain FedNova (equal) is already at the ceiling of what we found.** Nothing beats it by more than
  noise on both the notebook split and the 9 alternative splits.
- **A smaller server step (0.75× mean τ) is the only consistent refinement**, but it's tiny (+0.3 F1
  on average) and it adds a hyperparameter we'd have tuned against the test set. **We keep plain
  FedNova**: it has no tuned knobs, and its step length follows from the method rather than a search.
  If you asked us for "one more point", 0.75× is where it would come from.
- **SCAFFOLD confirms the diagnosis from a different direction.** Even with *size* weights, correcting
  each client's gradient toward the global direction takes F1 from 0.715 to 0.774 and gives the best
  AUC of any method (0.927). So client drift really is the mechanism. But SCAFFOLD needs a client-side
  change (control variates, and a second vector uploaded per round), while FedNova gets the same F1
  from the server alone. That's a good point for Q&A: "we tried the textbook drift fix, and a
  server-only fix matched it."
- **Fairness weights (q-FFL proxy) hurt F1 but keep AUC.** We measured the per-step update norms
  (seed 0, every round). Banks 0 and 1 have the largest (~0.007 by round 8), banks 3 and 4 are close
  behind (~0.006), and **bank 2 is by far the smallest (~0.0025)**. So norm-based weights mostly
  take the vote away from bank 2, the DoS-heavy bank (68% attacks). Our hypothesis is that this costs
  DoS knowledge, which fits the drop in recall (0.676 → 0.631 at q=2). We did not measure per-family
  recall for this variant.
- **Resolving conflicts (PCGrad) did nothing, even though there is plenty of conflict.** We measured
  cosine similarity between the banks' per-step directions (seed 0). Conflict starts near zero and
  **grows every round**. By round 8, the attack-rich banks (0, 1) point almost exactly *against* the
  normal-heavy banks (3, 4), with cosines of −0.86 to −0.91, and against bank 2 at about −0.55. Our
  interpretation: this conflict isn't noise to be removed. It *is* the disagreement about where the
  normal/attack boundary lies, so projecting it out removes the useful signal from both sides.
- **The growing conflict probably also explains the F1-over-rounds chart**, where FedNova is best
  after round 1 and eases down by round 8. As the global model settles, each group of banks mainly
  pushes back against the other's side of the boundary. This is a hypothesis we haven't tested.
- **Server momentum** again gives no gain. The remaining error isn't round-to-round noise.
