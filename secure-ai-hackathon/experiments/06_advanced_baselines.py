"""Stage 6: how do existing aggregators fare under the notebook's attack, and what do they cost
when nobody attacks? Notebook split, seeds 0-4."""
from adv_defenses import *

DEFENSES = {
    "FedAvg (no defense)": None,
    "Coordinate median (notebook)": median_(),
    "Norm clip 5 (notebook skeleton)": notebook_clip(),
    "Trimmed mean k=1": trimmed_mean(1),
    "Krum f=1": krum(1, 1),
    "Multi-Krum f=1 m=3": krum(1, 3),
    "Geometric median": geomed(),
    "FedNova equal (ours, undefended)": fednova_equal(),
}
SCENARIOS = {
    "clean": dict(malicious_clients=[]),
    "scale x15 [1]": dict(malicious_clients=[1], attack="scale", scale_factor=15.0),
    "label_flip [1]": dict(malicious_clients=[1], attack="label_flip"),
}
if __name__ == "__main__":
    for tag, atk in SCENARIOS.items():
        for name, fac in DEFENSES.items():
            adv_multi_seed(name, fac, tag=tag, **atk)
