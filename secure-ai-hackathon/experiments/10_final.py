"""Stage 10: final FedNova-Guard (kappa=5, clip=3, step=mean) on the notebook split, 5 seeds, the
scenarios not already covered by 08_stress.py."""
from adv_defenses import *
for tag, atk in {"label_flip [1]": dict(malicious_clients=[1], attack="label_flip"),
                 "norm_match [1]": dict(malicious_clients=[1], attack="norm_match")}.items():
    adv_multi_seed("FedNova-Guard final", fednova_guard(), tag=tag, log="advanced_stress", **atk)
