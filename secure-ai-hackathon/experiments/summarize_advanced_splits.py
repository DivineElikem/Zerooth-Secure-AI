"""Print per-split F1 / detection for the Advanced track (results/advanced_splits.jsonl)."""
import json, os, re
from collections import defaultdict
rows = [json.loads(l) for l in open(os.path.join(os.path.dirname(__file__), "results", "advanced_splits.jsonl"))]
T = defaultdict(dict)
for r in rows:
    split, name = re.match(r"\[(.*?)\] (.*)", r["name"]).groups()
    T[(r["tag"], name)][split] = r
splits = sorted({s for v in T.values() for s in v})
for tag in ["clean", "scale x15 [1]", "label_flip [1]", "norm_match [1]"]:
    print(f"\n=== {tag}   (F1 mean over training seeds; det = TP/FN/FP over rounds x seeds)")
    print(f"{'defense':32s}" + "".join(f"{s:>11s}" for s in splits) + "   worst gap-to-oracle")
    for name in ["FedAvg", "Coordinate median", "Multi-Krum f=1 m=3", "FedNova equal (undefended)",
                 "FedNova-Guard (ours)", "FedNova-Guard step=mean", "FedNova-Guard step=capmean", "Oracle: FedNova w/o attackers"]:
        v = T.get((tag, name))
        if not v:
            continue
        orc = T.get((tag, "Oracle: FedNova w/o attackers"), {})
        gaps = [v[s]["f1_mean"] - orc[s]["f1_mean"] for s in splits if s in orc and s in v]
        print(f"{name:32s}" + "".join(f"{v[s]['f1_mean']:11.4f}" if s in v else f"{'-':>11s}" for s in splits)
              + (f"   {min(gaps):+.4f}" if gaps else ""))
    g = T.get((tag, "FedNova-Guard (ours)"), {})
    print(f"{'  Guard det TP/FN/FP':32s}" + "".join(
        f"{'%d/%d/%d' % (g[s]['detection']['tp'], g[s]['detection']['fn'], g[s]['detection']['fp']) if s in g and g[s]['detection'] else '-':>11s}"
        for s in splits))
