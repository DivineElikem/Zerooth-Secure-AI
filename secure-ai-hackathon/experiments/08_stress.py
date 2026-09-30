"""Stage 8: stress matrix. Final FedNova-Guard (kappa=5, clip=3) vs baselines across attack scenarios.
Usage: python 08_stress.py <scenario-group 0..3>"""
import sys
from adv_defenses import *
torch.set_num_threads(3)

S = {
    "clean":                    dict(malicious_clients=[]),
    "scale x15 [1] (notebook)": dict(malicious_clients=[1], attack="scale", scale_factor=15.0),
    "scale x3 [1]":             dict(malicious_clients=[1], attack="scale", scale_factor=3.0),
    "scale x50 [1]":            dict(malicious_clients=[1], attack="scale", scale_factor=50.0),
    "scale x15 [1,3]":          dict(malicious_clients=[1, 3], attack="scale", scale_factor=15.0),
    "scale x50 [1,3]":          dict(malicious_clients=[1, 3], attack="scale", scale_factor=50.0),
    "label_flip [1,3]":         dict(malicious_clients=[1, 3], attack="label_flip"),
    "norm_match [1,3]":         dict(malicious_clients=[1, 3], attack="norm_match"),
    "scale x15 [1] from r4":    dict(malicious_clients=[1], attack="scale", scale_factor=15.0, start_round=4),
    "sign_flip x5 [1]":         dict(malicious_clients=[1], attack="sign_flip", scale_factor=5.0),
    "scale x15 [1] size=100":   dict(malicious_clients=[1], attack="scale", scale_factor=15.0, size_lie=100),
    "label_flip [1] size=1e6":  dict(malicious_clients=[1], attack="label_flip", size_lie=1_000_000),
}
GROUPS = [list(S)[i::4] for i in range(4)]

if __name__ == "__main__":
    for tag in GROUPS[int(sys.argv[1])]:
        atk = S[tag]
        D = {
            "FedAvg (no defense)": None,
            "Coordinate median": median_(),
            "Trimmed mean k=1": trimmed_mean(1),
            "Multi-Krum f=1 m=3": krum(1, 3),
            "Geometric median": geomed(),
            "FedNova equal (undefended)": fednova_equal(),
            "FedNova-Guard (ours)": fednova_guard(),
        }
        if atk["malicious_clients"]:
            D["Oracle: FedNova w/o attackers"] = fednova_oracle(tuple(atk["malicious_clients"]))
        for name, fac in D.items():
            adv_multi_seed(name, fac, tag=tag, log="advanced_stress", **atk)
