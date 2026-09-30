"""Stage 9c: Guard with step = mean tau capped at 25 x median tau (size-lie bound without slowing big honest banks)."""
import sys
from importlib import import_module
from adv_defenses import *
torch.set_num_threads(3)
s9, s8 = import_module("09_splits"), import_module("08_stress")
g = int(sys.argv[1])
if g < 4:
    for a, ps in s9.SPLITS[g::4]:
        clients = iid_clients if a == "iid" else make_noniid_partition(train_pool, alpha=a, seed=ps)
        split = "iid" if a == "iid" else f"a={a} s={ps}"
        for tag, atk in s9.SC.items():
            adv_multi_seed(f"[{split}] FedNova-Guard step=capmean", fednova_guard(step="capmean"), clients=clients,
                           seeds=[0, 1, 2], tag=tag, log="advanced_splits", **atk)
else:
    for tag, atk in s8.S.items():
        adv_multi_seed("FedNova-Guard step=capmean", fednova_guard(step="capmean"), tag=tag, log="advanced_stress", **atk)
