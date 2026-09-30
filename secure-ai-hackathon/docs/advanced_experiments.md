# Advanced track — experiment log

Working notes for the Advanced writeup: what we tried, what worked, what didn't, and why.
Raw numbers are in `experiments/results/advanced*.jsonl`. The Intermediate-track log
(`intermediate_experiments.md`) covers the non-IID diagnosis this builds on.

## Setup and rules

- **Attack (notebook):** bank 1 flips its labels and multiplies its update by 15, in every round.
  Bank 1 is small (10.7k rows), but it is also one of the two probe-heavy banks. Its *honest* data is
  some of the most valuable in the federation.
- **Score:** F1 on KDDTest+ at 0.5 after 8 rounds, same harness as the notebook. We also log AUC, and
  **detection** as counts over (bank, round) pairs: detected = malicious pairs flagged,
  false flags = honest pairs flagged.
- **Privacy rule (unchanged):** the server sees only each bank's model and its reported size.
- **Seeds:** 5 training seeds (0–4) on the notebook split; 3 seeds on the 9 alternative splits.
- **Reference points:** our Intermediate method (FedNova, equal weights) scores 0.7787 with no attack.
  The **oracle** is FedNova that is simply told which banks are malicious and drops them in every
  round. It is the ceiling for any defense, because a perfect defense also loses the attacker's
  honest data.
- **Code:** `experiments/adv_common.py` is an attack-aware copy of the notebook's `run_fl`. It
  reproduces `run_fl` bit-for-bit (checked on seeds for both `scale` and `label_flip`) and adds:
  an attack start round; a norm-matched label flip (the flipped update is rescaled to the size the
  bank's honest update would have had, to evade norm checks); sign flip; and lying about dataset size.
  `experiments/adv_defenses.py` holds every aggregator.

## Stage 6 — baselines (`06_advanced_baselines.py`)

Notebook split, 5 seeds, F1:

| Aggregation | No attack | Scale ×15 (notebook) | Label flip only |
|---|---|---|---|
| Naive FedAvg | 0.7147 | **0.2462** | 0.6878 |
| Coordinate median (notebook floor) | 0.7288 | 0.6804 | 0.6771 |
| Norm clip at 5 (notebook skeleton) | 0.7147 | 0.6128 | 0.6878 |
| Trimmed mean, k=1 | 0.7452 | 0.6909 | 0.6911 |
| Krum, f=1 | 0.6637 | 0.6586 | 0.6586 |
| Multi-Krum, f=1, keeps 3 | 0.7198 | 0.6973 | 0.7081 |
| Geometric median | 0.7292 | 0.6746 | 0.6742 |
| **FedNova equal (our Intermediate method)** | **0.7787** | **0.2241** (one seed at 0.000) | 0.6924 |

- **Our Intermediate winner is the most fragile method under attack.** Equal weights hand the
  attacker 20% of the vote whatever its size. FedNova then divides by its (small) step count, which
  makes its per-step update larger still.
- **Every standard defense has a cost when nobody attacks.** Krum drops to 0.664 without any attack
  and flags honest banks 160 times in 40 bank-rounds. They pay this cost because under non-IID data,
  the "odd one out" is usually an honest bank with rare knowledge.

## Diagnosis — what can a server actually see?

We traced three statistics per bank and per round: per-step update norm relative to the median,
cosine to the coordinate-wise median, and the maximum cosine to any other bank. We traced them both
with no attack and along the *perfectly defended* trajectory (bank 1 dropped), since that is the
trajectory a working defense will be on.

**1. Direction can't separate an attacker from an honest minority bank.** Once bank 1 is dropped,
honest bank 0 becomes the only probe expert and the most anomalous-looking bank by direction:
cosine to the median −0.67, and not a single bank it agrees with (max cosine −0.33). The
*norm-matched* label flipper looks *more* ordinary (cosine to the median +0.5). The reason is that
flipped probe labels say "probes are normal", which agrees with the three normal-heavy banks. Krum
and cosine filters therefore remove bank 0.

**2. Size per SGD step does separate them.** Dividing each update by the bank's step count
τᵢ = ⌈nᵢ/256⌉ (which FedNova already does), the ratio to the median is:

| | Honest banks | Scaled attacker | Plain label flip |
|---|---|---|---|
| Per-step norm / median | ≤ 1.7 (no attack), ≤ 2.7 (any trace) | 19 → 148 | 1.3 → 9.9, growing each round |

The plain label flip *grows* because the better the model gets, the more it contradicts flipped labels.

## Stage 7 — FedNova-Guard, and the "framing" trap (`07_guard_tune.py`, `07b_guard_grid.py`)

**The rule, applied each round:**
1. Compute FedNova per-step updates dᵢ = Δᵢ/τᵢ, and m = the median of ‖dᵢ‖.
2. **Detect:** if ‖dᵢ‖ > κ·m, exclude bank i this round. After 2 flags it is quarantined for good.
3. **Bound:** clip the remaining banks to ‖dᵢ‖ ≤ clip·m.
4. **Aggregate:** FedNova with equal weights over the remaining banks.

**First try (κ=3, clip 2): worked on the scale attack, failed on label flip** (0.679, with 30 false
flags). The trace showed a mechanism we call **framing**:

| Round | Bank 0 (honest) | Bank 1 (flipper) | Flagged |
|---|---|---|---|
| 1 | 1.59 | 1.27 | – |
| 2 | 2.39 | 1.92 | – |
| 3 | **3.10** | 2.88 | **bank 0** |
| 4 | 5.18 | 3.26 | banks 0, 1 |
| 8 | 10.08 | 13.87 | banks 0, 1 (both quarantined) |

While the flipper is still under the threshold, it drags the model toward "probes are normal". The
only bank that disagrees strongly is bank 0, the other probe expert. Its corrective update grows and
crosses the threshold *first*. Once bank 0 is excluded, the model drifts further from its data, so it
looks even more anomalous. The attacker gets the honest expert removed. The underlying ambiguity
(two banks pulling hard in opposite directions) is one no updates-only server can resolve.

**Grid (κ × clip), judged by the worst case over attack types, not the best single score.** Notebook
split, 5 seeds; FP = false flags:

| κ | clip | No attack | Scale ×15 | Label flip | Norm-matched | Worst |
|---|---|---|---|---|---|---|
| 3 | 2 | 0.7785 | 0.7579 | 0.6787 (FP 30) | 0.6791 (FP 29) | 0.679 |
| 4 | 2 | 0.7785 | 0.7579 | 0.7464 | 0.6913 | 0.691 |
| 5 | 2 | 0.7785 | 0.7579 | 0.7465 | 0.7011 | 0.701 |
| 5 | 2.5 | 0.7785 | 0.7645 | 0.7554 | 0.7219 | 0.722 |
| **5** | **3** | 0.7785 | 0.7645 | 0.7346 | 0.7389 | **0.735** |
| 5 | none | 0.7785 | 0.7645 | 0.6923 | 0.7389 | 0.692 |
| ∞ (clip only) | 2 | 0.7785 | 0.7013 | 0.7009 | 0.7011 | 0.701 |
| ∞ (no defense) | none | 0.7785 | 0.4436 | 0.6923 | 0.7389 | 0.444 |

- **Tighter thresholds are worse, not safer.** κ=3 walks into the framing trap.
- **Clipping alone holds every attack to ~0.70.** Detection is what lifts the notebook attack to the oracle.
- **Choice: κ=5, clip=3.** Clip 3 also has a reason that doesn't come from test scores: the largest
  honest ratio in any trace was 2.7, so a 3× clip is the tightest bound that never clips an honest
  bank. Clip 2.5 is within noise of it.

## Stage 8 — stress matrix (`08_stress.py`)

Notebook split, 5 seeds, F1. "Guard" is the final configuration (κ=5, clip 3, mean τ; see stage 9).

| Scenario | FedAvg | Median | Trimmed mean | Multi-Krum | Geo. median | FedNova (undef.) | **Guard** | Oracle | Guard detected / false flags |
|---|---|---|---|---|---|---|---|---|---|
| No attack | 0.715 | 0.729 | 0.745 | 0.720 | 0.729 | 0.779 | **0.779** | – | – / 0 |
| Scale ×15, bank 1 (notebook) | 0.246 | 0.680 | 0.691 | 0.697 | 0.675 | 0.224 | **0.764** | 0.764 | 40/40 / 0 |
| Scale ×3, bank 1 | 0.659 | 0.678 | 0.692 | 0.697 | 0.674 | 0.649 | **0.764** | 0.764 | 31/40 / 0 |
| Scale ×50, bank 1 | 0.533 | 0.680 | 0.694 | 0.697 | 0.675 | 0.565 | **0.764** | 0.764 | 40/40 / 0 |
| Scale ×15, banks 1+3 | 0.198 | 0.746 | 0.448 | 0.736 | 0.745 | 0.497 | **0.774** | 0.774 | 80/80 / 0 |
| Scale ×50, banks 1+3 | 0.463 | 0.740 | 0.485 | 0.736 | 0.745 | 0.612 | **0.774** | 0.774 | 80/80 / 0 |
| Scale ×15 from round 4 ("partway") | 0.523 | 0.700 | 0.702 | 0.711 | 0.690 | 0.381 | **0.765** | 0.764 | 20/20 / 0 |
| Sign flip ×5, bank 1 | 0.415 | 0.667 | 0.695 | 0.697 | 0.667 | **0.000** | **0.764** | 0.764 | 40/40 / 0 |
| Scale ×15, claims 100 rows | 0.704 | 0.680 | 0.691 | 0.697 | 0.675 | 0.629 | **0.764** | 0.764 | 40/40 / 0 |
| Label flip, bank 1 | 0.688 | 0.677 | 0.691 | 0.708 | 0.674 | 0.692 | **0.735** | 0.764 | 1/40 / 0 |
| Norm-matched flip, bank 1 | – | – | – | – | – | – | **0.739** | 0.763 | 0/40 / 0 |
| Label flip, banks 1+3 | 0.721 | 0.751 | 0.713 | 0.620 | 0.745 | 0.779 | 0.779 | 0.774 | 0/80 / 0 |
| Norm-matched flip, banks 1+3 | 0.705 | 0.757 | 0.736 | 0.581 | 0.750 | 0.792 | 0.792 | 0.774 | 0/80 / 0 |
| Label flip, claims 1,000,000 rows | 0.208 | 0.677 | 0.691 | 0.708 | 0.674 | 0.678 | **0.715** | 0.764 | 0/40 / 9 |

**Findings**

- **Against the notebook's attack, and every attack that inflates the update, Guard equals the
  oracle.** That covers scale ×3 to ×50, two attackers, a late start, sign flip, and a small-size lie.
  Detection is perfect with zero false flags. At ×3 it misses 9 of 40 early bank-rounds, where the
  update is still under 5×; clipping covers those rounds.
- **Zero false flags in every scenario except the size lie.**
- **Unscaled flips are bounded, not detected.** Guard beats every baseline on them but stays about
  0.03 under the oracle (see diagnosis point 1 and the framing section).
- **Two attackers flipping labels (banks 1+3) "raises" F1 above the oracle, but AUC drops** (0.877 vs
  0.918). Bank 3 is normal-heavy, so flipping its labels pushes the model toward predicting
  "attack". On a 57%-attack test set that looks like better F1 at a fixed 0.5 threshold. This is the
  same threshold-shift trap as Intermediate stage 4; it is not a real gain.

## Stage 9 — does it generalise? 9 other splits and IID (`09_splits.py`, `09b_step_mean.py`, `09c_capmean.py`)

Dirichlet α ∈ {0.1, 0.3, 1.0} × partition seeds {1, 2, 3}, plus IID. Bank 1 is the attacker,
whatever data it happens to hold. 3 seeds each. `summarize_advanced_splits.py` prints the full tables.

**Server step: median τ vs mean τ.** Our first Guard used the **median** τ as the step length, so that a
bank lying about its size couldn't inflate the step. On the splits, that choice backfired. On α=0.1
split 1, one honest bank holds 98k of 111k rows and is the only real source of normal traffic. With
the median-τ step, it can't pull the model back from "everything is an attack" within 8 rounds (F1
0.726 vs 0.837 for the oracle), even though the attacker is caught in every round. The same happens
with no attack at all on α=0.1 split 3 (0.723 vs 0.843). **Switching to the mean τ (plain FedNova)
fixed it:**

| Worst gap to the oracle across the 10 splits | Median-τ Guard | **Mean-τ Guard (final)** |
|---|---|---|
| Scale ×15 | −0.112 | **−0.017** |
| Label flip | −0.112 | −0.112 |
| Norm-matched flip | −0.113 | **−0.058** |

The cost is the size lie. Claiming 1,000,000 rows inflates the mean τ: F1 0.715 ± 0.07, with 9 false
flags. With the median τ it was 0.765. We also tried capping each bank's τ at 25× the median τ. It
was identical on honest splits and *worse* under the lie (0.643), so we dropped it.

**Final Guard on the splits:**
- **Scale ×15:** detected in 236 of 240 attacker bank-rounds, and within 0.017 of the oracle
  everywhere.
- **False flags:** none at α=0.3, α=1.0 or IID. **Under extreme skew (α=0.1) there are some:** 18 and
  8 with no attack on splits 1 and 2, and 19 under attack on split 3. In those splits some honest banks
  are tiny and nearly 100% attack traffic, so they really do move more than 5× the median per step. It
  costs no F1 there (0.830 vs 0.818 undefended; 0.809 vs 0.811). Split 3 is degenerate: every method,
  the oracle included, ends at 0.7253, which is predicting "attack" for everything.

## Final choice: FedNova-Guard (κ=5, clip=3, mean τ, 2 strikes)

The same server-side information as FedAvg (models and sizes), no extra client protocol, and no
change at all when nobody attacks.

**Headline (the notebook's own run: seed 42, the notebook split, the notebook attack; every run seeded
identically), from `02_Day2_Intermediate_Advanced.ipynb`:**

| | F1 |
|---|---|
| No attack, naive FedAvg | 0.7117 |
| **Under attack, naive FedAvg** | **0.0511** |
| Under attack, coordinate median (notebook floor) | 0.6855 |
| **Under attack, FedNova-Guard** | **0.7760** (P 0.925 / R 0.668) |
| Recovered vs naive FedAvg under attack | **+0.7249** |

Detection: bank 1 was flagged in all 8 rounds, with no honest bank ever flagged. The defended model
under attack scores *above* naive FedAvg with no attack, because it also carries the Intermediate
fix. Exported model (`submission_advanced/`): AUC 0.900, false-positive rate 7.2%, probe recall 0.93,
R2L/U2R recall 0.17. It was checked in a fresh process with `torch.jit.load` and matches
`submission.json`.

**Supporting numbers (5 seeds):** no attack 0.7787 (same as undefended FedNova); notebook attack
0.7639 ± 0.0093 (oracle 0.7639; FedAvg 0.2462; median 0.6804), with 40/40 detected and 0 false flags.

## Limitations (for the writeup and Q&A)

- **A stealthy flipper is bounded, not caught.** A label flipper that keeps its update at an honest
  size costs about 0.03 F1 against the oracle. Our diagnosis argues that no updates-only server can
  reliably tell such an attacker from an honest minority bank under non-IID data. Resolving it needs
  information we deliberately don't use. A small trusted validation set on the server (FLTrust-style)
  is the natural next step, but it's a new trust assumption.
- **Size lies.** Reported sizes are unverifiable. The mean-τ step can be inflated by a bank claiming
  a huge dataset (0.715). Median τ closes the hole but stalls federations dominated by one honest
  bank. We chose the option that is better on honest federations, and we say so.
- **Extreme skew causes false flags** (α=0.1 only), with no measured F1 cost.
- **Excluding the attacker also loses its honest data.** That's why the oracle (0.764) sits below the
  no-attack FedNova number (0.779). It's the price of a compromised bank, not of the defense.
