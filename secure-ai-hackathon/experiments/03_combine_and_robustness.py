"""Stage 3: combine the stage-2 winners, then check they generalise beyond the notebook's
single split (other Dirichlet alphas / partition seeds, IID, longer training)."""
import sys
from common import *
from importlib import import_module
s2 = import_module("02_privacy_safe")


def fednova_dissim(temp=3.0):
    """FedNova per-step directions, weighted by exp(-temp * mean cosine to other clients)."""
    def factory():
        def agg(lm, sizes, prev):
            n = np.array(sizes, float)
            tau = np.ceil(n / s2.BATCH)
            d = s2.deltas_of(lm, prev) / torch.tensor(tau, dtype=torch.float32).unsqueeze(1)
            u = d / d.norm(dim=1, keepdim=True)
            k = len(lm)
            mean_cos = (((u @ u.T).sum(1) - 1) / (k - 1)).numpy()
            p = np.exp(-temp * mean_cos); p = p / p.sum()
            tau_eff = float((p * tau).sum())
            return model_from_flat(prev, get_flat_params(prev) + tau_eff * weighted_mean(d, p))
        return active_only(agg)
    return factory


def fednova_gamma(gamma):
    """FedNova with client weights p_i proportional to n_i**gamma (0 = equal, 1 = size)."""
    def factory():
        def agg(lm, sizes, prev):
            n = np.array(sizes, float)
            tau = np.ceil(n / s2.BATCH)
            p = n ** gamma; p = p / p.sum()
            d = s2.deltas_of(lm, prev) / torch.tensor(tau, dtype=torch.float32).unsqueeze(1)
            return model_from_flat(prev, get_flat_params(prev) + float((p * tau).sum()) * weighted_mean(d, p))
        return active_only(agg)
    return factory


CANDIDATES = {
    "naive FedAvg": None,
    "equal weights": s2.EQUAL,
    "FedNova (equal)": s2.fednova("equal"),
    "dissimilarity T=3": s2.dissimilarity_weights(3.0),
    "FedNova + dissim T=3": fednova_dissim(3.0),
    "FedNova + dissim T=5": fednova_dissim(5.0),
}

GROUPS = {
    "combine": lambda: [
        multi_seed("dissimilarity T=5", s2.dissimilarity_weights(5.0)),
        multi_seed("dissimilarity T=10", s2.dissimilarity_weights(10.0)),
        multi_seed("FedNova + dissim T=1", fednova_dissim(1.0)),
        multi_seed("FedNova + dissim T=3", fednova_dissim(3.0)),
        multi_seed("FedNova + dissim T=5", fednova_dissim(5.0)),
    ],
    "splits": lambda: [
        multi_seed(f"[alpha={a} split={ps}] {name}", f,
                   clients=make_noniid_partition(train_pool, alpha=a, seed=ps),
                   seeds=[0, 1, 2], log="intermediate_splits")
        for a in [0.1, 0.3, 1.0] for ps in [1, 2, 3] for name, f in CANDIDATES.items()
    ],
    "empty_client_rerun": lambda: [  # alpha=0.1 split=2 has a bank with 0 rows; re-run after adding active_only
        multi_seed(f"[alpha=0.1 split=2] {name}", f, clients=make_noniid_partition(train_pool, alpha=0.1, seed=2),
                   seeds=[0, 1, 2], log="intermediate_splits")
        for name, f in CANDIDATES.items()
    ],
    "gamma": lambda: [multi_seed(f"FedNova, weights ~ size^{g}", fednova_gamma(g)) for g in [0.0, 0.25, 0.5, 0.75, 1.0]],
    "iid": lambda: [multi_seed(f"[IID] {name}", f, clients=iid_clients) for name, f in CANDIDATES.items()],
    "rounds": lambda: [multi_seed(f"[20 rounds] {name}", f, rounds=20) for name, f in CANDIDATES.items()],
}

if __name__ == "__main__":
    for g in (sys.argv[1:] or GROUPS):
        print(f"--- {g} ---"); GROUPS[g]()
