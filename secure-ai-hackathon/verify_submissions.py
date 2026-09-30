"""Check that each exported model reproduces the metrics in its submission.json.

    python verify_submissions.py

Loads submission_<track>/model_scripted.pt with torch.jit.load, scores it on KDDTest+ with the
notebook's own preprocessing (experiments/harness.py is a verbatim copy of the notebook's setup
cells), and compares precision / recall / F1 with the self-reported numbers.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "experiments"))

import torch  # noqa: E402
from harness import evaluate  # noqa: E402

ok = True
for track in ["intermediate", "advanced"]:
    folder = os.path.join(ROOT, f"submission_{track}")
    reported = json.load(open(os.path.join(folder, "submission.json")))
    model = torch.jit.load(os.path.join(folder, reported["model_file"]))
    model.eval()
    got = evaluate(model)
    match = got == reported["self_reported_metrics"]
    ok &= match
    print(f"\n{track:>12}: reported {reported['self_reported_metrics']}")
    print(f"{'':>12}  recomputed {got}  ->  {'MATCH' if match else 'MISMATCH'}")

sys.exit(0 if ok else 1)
