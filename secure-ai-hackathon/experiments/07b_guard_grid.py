"""Stage 7b: FedNova-Guard kappa x clip grid, judged by worst case over attack types."""
import sys
from adv_defenses import *
torch.set_num_threads(3)
SC = {
    "clean": dict(malicious_clients=[]),
    "scale x15 [1]": dict(malicious_clients=[1], attack="scale", scale_factor=15.0),
    "label_flip [1]": dict(malicious_clients=[1], attack="label_flip"),
    "norm_match [1]": dict(malicious_clients=[1], attack="norm_match"),
}
KAPPAS = {"3": 3.0, "4": 4.0, "5": 5.0, "inf": float("inf")}
if __name__ == "__main__":
    k = KAPPAS[sys.argv[1]]
    for clip in [1.0, 1.5, 2.0, None]:
        for tag, atk in SC.items():
            adv_multi_seed(f"Guard kappa={sys.argv[1]} clip={clip}", fednova_guard(kappa=k, clip=clip),
                           tag=tag, log="advanced_grid", **atk)
