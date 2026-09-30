"""Stage 9b: Guard with server step = mean tau (as in FedNova) instead of median tau.
Median tau resists size inflation lies; mean tau lets one very large honest bank move the model faster."""
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
            adv_multi_seed(f"[{split}] FedNova-Guard step=mean", fednova_guard(step="mean"), clients=clients,
                           seeds=[0, 1, 2], tag=tag, log="advanced_splits", **atk)
else:
    for tag, atk in s8.S.items():
        adv_multi_seed("FedNova-Guard step=mean", fednova_guard(step="mean"), tag=tag, log="advanced_stress", **atk)
