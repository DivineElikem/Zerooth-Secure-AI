"""Stage 7a: FedNova-Guard vs the oracle, clean and under the notebook attacks; knob sensitivity."""
from adv_defenses import *
SC = {
    "clean": dict(malicious_clients=[]),
    "scale x15 [1]": dict(malicious_clients=[1], attack="scale", scale_factor=15.0),
    "label_flip [1]": dict(malicious_clients=[1], attack="label_flip"),
    "norm_match [1]": dict(malicious_clients=[1], attack="norm_match"),
}
if __name__ == "__main__":
    for tag, atk in SC.items():
        if tag != "clean":
            adv_multi_seed("Oracle: FedNova w/o bank 1", fednova_oracle(), tag=tag, **atk)
        for kw in [dict(), dict(clip=None), dict(clip=1.5), dict(clip=3.0), dict(kappa=2.5), dict(kappa=4.0), dict(step="mean")]:
            adv_multi_seed(f"Guard {kw or 'default'}", fednova_guard(**kw), tag=tag, **atk)
