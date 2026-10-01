# Team zerooth: CAIRLab Secure AI Hackathon, Day 2 (Intermediate + Advanced)

Federated intrusion detection on NSL-KDD across five banks. We submit to both Day 2 tracks:

| Track | Problem | Our method | Headline F1 (seed 42, notebook split) |
|---|---|---|---|
| 🟡 **Intermediate** | Bank data is skewed (non-IID), and naive FedAvg loses accuracy | **FedNova with equal client weights** | 0.7117 → **0.7821** (+0.070) |
| 🔴 **Advanced** | Bank 1 is malicious (label flip + update ×15 every round) | **FedNova-Guard**: detect by per-step update size, clip, then FedNova-average | 0.0511 → **0.7760** (+0.725) |

Both methods run on the server and use only what naive FedAvg already receives: each bank's
trained model and its reported dataset size. No labels, histograms or other statistics leave a bank.

## Submission files (for verification)

All the code is in one notebook, [`02_Day2_Intermediate_Advanced.ipynb`](02_Day2_Intermediate_Advanced.ipynb). Its 🟡 Intermediate and 🔴 Advanced sections sit where the
organisers' template puts them, and its final cell exports both tracks. Each track has its own folder
with the two files the organisers asked for:

| Track | Model | Metrics | Produced by |
|---|---|---|---|
| Intermediate | [`submission_intermediate/model_scripted.pt`](submission_intermediate/model_scripted.pt) | [`submission_intermediate/submission.json`](submission_intermediate/submission.json) | [`02_Day2_Intermediate_Advanced.ipynb`](02_Day2_Intermediate_Advanced.ipynb), Section 8 |
| Advanced | [`submission_advanced/model_scripted.pt`](submission_advanced/model_scripted.pt) | [`submission_advanced/submission.json`](submission_advanced/submission.json) | [`02_Day2_Intermediate_Advanced.ipynb`](02_Day2_Intermediate_Advanced.ipynb), Section 8 |

```
intermediate  P 0.9173  R 0.6816  F1 0.7821   (41 input features)
advanced      P 0.9250  R 0.6684  F1 0.7760   (41 input features, trained under the notebook's attack)
```

Both models take the notebook's preprocessed 41-feature input (label-encoded categoricals,
`StandardScaler` fitted on KDDTrain+) and output one logit. The threshold is 0.5 after a sigmoid.

**One-command check.** This loads each `model_scripted.pt` with `torch.jit.load`, scores it on
KDDTest+ with the notebook's preprocessing, and compares the result with `submission.json`:

```bash
python verify_submissions.py
#  intermediate: ... -> MATCH
#      advanced: ... -> MATCH
```

## Setup

```bash
git clone <this repo> && cd <repo>
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Everything runs on CPU. The notebook downloads NSL-KDD (`KDDTrain+` / `KDDTest+`) from the same
GitHub mirror as the organisers' notebook into `data/`. The experiment scripts use a cached
copy in `experiments/data/`.

## Reproducing the results

One command runs both tracks, top to bottom (about 4 minutes on a laptop CPU):

```bash
jupyter nbconvert --to notebook --execute --inplace 02_Day2_Intermediate_Advanced.ipynb
```

It regenerates `submission_intermediate/`, `submission_advanced/` and both charts in `figures/`.
Re-running gives the same precision, recall and F1 as the committed `submission.json` files.
The weights can differ by about 1e-7 (float32 rounding on CPU).

### Intermediate track (🟡 section of the notebook)

Every
federated run is seeded right before `run_fl` (`SEED = 42`, the notebook's own seed). This means
the before/after comparison uses identical randomness, and the exported model's F1 is the reported
number. The notebook also prints a 5-seed comparison (seeds 0–4):

| Non-IID, 8 rounds, 5 seeds | F1 |
|---|---|
| IID FedAvg (reference) | 0.7556 ± 0.0064 |
| Naive FedAvg | 0.7147 ± 0.0086 |
| **FedNova, equal weights** | **0.7787 ± 0.0082** (wins on every seed) |

**Why it works:** under skew, size-weighted FedAvg counts a big bank twice. It gets a bigger
weight, and it also sends a longer update, because it takes more local SGD steps. FedNova divides
out the step count (τᵢ = ⌈nᵢ / 256⌉), and equal weights remove the size effect. The small banks'
probe and rare-attack knowledge then survives: probe recall rises from 0.61 to 0.91 and AUC from
0.877 to 0.920. The cost is a false-positive rate of 7.9% vs 3.9%. The method also beats naive
FedAvg on 9 of 9 alternative Dirichlet splits.

### Advanced track (🔴 section of the notebook)

The Guard builds directly on the Intermediate method (FedNova, equal weights).

**FedNova-Guard, applied every round** (κ = 5, clip = 3, two strikes):

1. **Normalise:** compute each bank's per-step update dᵢ = Δᵢ / τᵢ and the median norm m.
2. **Detect:** if ‖dᵢ‖ > 5·m, exclude bank i this round. After 2 flags it is quarantined for good.
3. **Bound:** clip the remaining banks to ‖dᵢ‖ ≤ 3·m. No honest bank exceeded 2.7·m in any trace,
   so honest banks are never clipped.
4. **Aggregate:** FedNova with equal weights over the banks that remain.

| Notebook attack, 5 seeds | F1 |
|---|---|
| Naive FedAvg | 0.2462 |
| Coordinate median (notebook's baseline defense) | 0.6804 |
| FedNova, equal weights, undefended | 0.2241 |
| **FedNova-Guard** | **0.7639 ± 0.0093**: 40/40 attacker bank-rounds detected, 0 false flags |
| Oracle (told which bank is malicious) | 0.7639 |
| *No attack: FedNova-Guard / FedNova* | *0.7787 / 0.7787 (no cost when nobody attacks)* |

The Guard matches the oracle against every attack that inflates the update: scale ×3 to ×50, two
attackers, a late start, sign flip, and a bank that lies about its size. It beats every baseline
on stealthy (norm-matched) label flips, but sits about 0.03 below the oracle there. See
*Limitations* in the Advanced log.

## Experiment scripts (full evidence, including what failed)

The notebook shows the final methods. The search that led to them is in `experiments/`, and
every result is logged to `experiments/results/*.jsonl`. The write-ups with every table:
[`docs/intermediate_experiments.md`](docs/intermediate_experiments.md) and
[`docs/advanced_experiments.md`](docs/advanced_experiments.md).

`experiments/harness.py` is a verbatim copy of the notebook's setup cells (data, partitions, model,
`run_fl`), so the scripts run outside Jupyter on exactly the same pipeline.
`experiments/adv_common.py` reproduces the notebook's attacked `run_fl` bit-for-bit and adds the
extra attacks.

```bash
cd experiments
python 01_baselines.py            # run a stage
python 08_stress.py 0             # scripts that take a number split their work into groups 0-3
python summarize_splits.py        # print summary tables from results/
python summarize_advanced_splits.py
```

| Script | Track | What it tests |
|---|---|---|
| `01_baselines.py` | Intermediate | Reference points and simple size re-weightings |
| `02_privacy_safe.py` | Intermediate | FedNova, FedProx, local-step caps, FedAdam/Yogi, dissimilarity weights |
| `03_combine_and_robustness.py` | Intermediate | Combining the winners; 9 alternative splits, IID, 20 rounds |
| `04_diagnostics.py` | Intermediate | Real gain or threshold shift? AUC, per-family recall, holdout |
| `05_round2.py` | Intermediate | Can anything beat FedNova (equal)? |
| `06_advanced_baselines.py` | Advanced | FedAvg, median, trimmed mean, Krum, Multi-Krum and geometric median, under attack and clean |
| `07_guard_tune.py`, `07b_guard_grid.py <3\|4\|5\|inf>` | Advanced | Guard vs oracle; κ × clip grid judged by worst case |
| `08_stress.py <0-3>` | Advanced | Stress matrix: attack strengths, two attackers, late start, sign flip, size lies |
| `09_splits.py <0-3>`, `09b_step_mean.py <0-4>`, `09c_capmean.py <0-4>` | Advanced | Generalisation to 9 splits + IID; median vs mean τ |
| `10_final.py` | Advanced | Final configuration, remaining scenarios |

## Repository layout

```
02_Day2_Intermediate_Advanced.ipynb  the submission notebook: both tracks, fully run, exports both submissions
submission_intermediate/             model_scripted.pt + submission.json (Intermediate)
submission_advanced/                 model_scripted.pt + submission.json (Advanced)
verify_submissions.py                re-scores both exported models against their submission.json
docs/                                experiment logs for both tracks
experiments/                         research scripts, shared helpers, cached data, raw results
figures/                             charts used in the write-up; make_guard_animation.py renders the "how it works" video
```

In the notebook, everything above Section 5 is the organisers' code, unchanged. Our additions are
the 🔧 cells, the "Our approach / Our defense" sections after them, and the note at the top.
