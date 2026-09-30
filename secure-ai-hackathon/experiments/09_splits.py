"""Stage 9: does FedNova-Guard generalise? 9 alternative non-IID splits (alpha x partition seed)
plus IID, training seeds 0-2. Client 1 is malicious whatever data it happens to hold.
Usage: python 09_splits.py <group 0..3>"""
import sys
from adv_defenses import *
torch.set_num_threads(3)

SPLITS = [(a, ps) for a in [0.1, 0.3, 1.0] for ps in [1, 2, 3]] + [("iid", None)]
SC = {
    "clean": dict(malicious_clients=[]),
    "scale x15 [1]": dict(malicious_clients=[1], attack="scale", scale_factor=15.0),
    "label_flip [1]": dict(malicious_clients=[1], attack="label_flip"),
    "norm_match [1]": dict(malicious_clients=[1], attack="norm_match"),
}
if __name__ == "__main__":
    for a, ps in SPLITS[int(sys.argv[1])::4]:
        clients = iid_clients if a == "iid" else make_noniid_partition(train_pool, alpha=a, seed=ps)
        split = "iid" if a == "iid" else f"a={a} s={ps}"
        for tag, atk in SC.items():
            D = {"FedAvg": None, "Coordinate median": median_(), "Multi-Krum f=1 m=3": krum(1, 3),
                 "FedNova equal (undefended)": fednova_equal(), "FedNova-Guard (ours)": fednova_guard()}
            if atk["malicious_clients"]:
                D["Oracle: FedNova w/o attackers"] = fednova_oracle((1,))
            for name, fac in D.items():
                adv_multi_seed(f"[{split}] {name}", fac, clients=clients, seeds=[0, 1, 2],
                               tag=tag, log="advanced_splits", **atk)
