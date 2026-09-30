"""Stage 2: privacy-safe improvements. Server knowledge: client models + sizes only.
Local steps per round are derivable from size (ceil(n / batch_size)), so FedNova is allowed."""
import sys
from common import *

BATCH = 256


def deltas_of(local_models, prev):
    return flats_of(local_models) - get_flat_params(prev)


# ---- drift correction -------------------------------------------------------------
def fednova(weighting="size"):
    """FedNova (Wang et al. 2020) for plain SGD: divide each client's update by its number
    of local steps tau_i, average the per-step directions, rescale by the effective tau."""
    def factory():
        def agg(lm, sizes, prev):
            n = np.array(sizes, float)
            tau = np.ceil(n / BATCH)
            p = n / n.sum() if weighting == "size" else np.ones_like(n) / len(n)
            d = deltas_of(lm, prev) / torch.tensor(tau, dtype=torch.float32).unsqueeze(1)
            tau_eff = float((p * tau).sum())
            return model_from_flat(prev, get_flat_params(prev) + tau_eff * weighted_mean(d, p))
        return active_only(agg)
    return factory


# ---- server optimizers ------------------------------------------------------------
def server_opt(kind="adam", server_lr=0.1, b1=0.9, b2=0.99, eps=1e-3, weighting="size"):
    """FedAdam / FedYogi (Reddi et al. 2021): treat the averaged delta as a pseudo-gradient."""
    def factory():
        st = {"m": None, "v": None}
        def agg(lm, sizes, prev):
            n = np.array(sizes, float)
            w = n if weighting == "size" else np.ones_like(n)
            g = weighted_mean(deltas_of(lm, prev), w)
            if st["m"] is None:
                st["m"] = torch.zeros_like(g); st["v"] = torch.full_like(g, eps ** 2)
            st["m"] = b1 * st["m"] + (1 - b1) * g
            if kind == "adam":
                st["v"] = b2 * st["v"] + (1 - b2) * g ** 2
            else:  # yogi
                st["v"] = st["v"] - (1 - b2) * g ** 2 * torch.sign(st["v"] - g ** 2)
            return model_from_flat(prev, get_flat_params(prev) + server_lr * st["m"] / (st["v"].sqrt() + eps))
        return active_only(agg)
    return factory


# ---- update-based weighting -------------------------------------------------------
def normalized_direction(scale="mean"):
    """Each client contributes a unit-norm direction (equal vote on *where* to move);
    step length is the mean (or median) client update norm."""
    def factory():
        def agg(lm, sizes, prev):
            d = deltas_of(lm, prev)
            norms = d.norm(dim=1, keepdim=True)
            step = norms.mean() if scale == "mean" else norms.median()
            direction = (d / norms).mean(0)
            direction = direction / direction.norm()
            return model_from_flat(prev, get_flat_params(prev) + step * direction)
        return active_only(agg)
    return factory


def dissimilarity_weights(temp=1.0):
    """Up-weight clients whose update is *least* like the others (they carry knowledge the
    majority doesn't have). Weight = exp(-temp * mean cosine to the other clients)."""
    def factory():
        def agg(lm, sizes, prev):
            d = deltas_of(lm, prev)
            u = d / d.norm(dim=1, keepdim=True)
            cos = u @ u.T
            k = len(lm)
            mean_cos = ((cos.sum(1) - 1) / (k - 1)).numpy()
            return model_from_flat(prev, get_flat_params(prev) + weighted_mean(d, np.exp(-temp * mean_cos)))
        return active_only(agg)
    return factory


EQUAL = lambda: active_only(lambda lm, sizes, prev: model_from_flat(prev, weighted_mean(flats_of(lm), np.ones(len(lm)))))

GROUPS = {
    "drift": lambda: [
        multi_seed("FedNova (size weights)", fednova("size")),
        multi_seed("FedNova (equal weights)", fednova("equal")),
        multi_seed("FedProx mu=0.01 + size", prox_mu=0.01),
        multi_seed("FedProx mu=0.1 + size", prox_mu=0.1),
        multi_seed("FedProx mu=1.0 + size", prox_mu=1.0),
        multi_seed("FedProx mu=0.1 + equal", EQUAL, prox_mu=0.1),
        multi_seed("step cap 25/round + size", max_steps=25),
        multi_seed("step cap 25/round + equal", EQUAL, max_steps=25),
    ],
    "serveropt": lambda: [
        multi_seed(f"Fed{k.capitalize()} lr={lr} + {w}", server_opt(k, lr, weighting=w))
        for k in ["adam", "yogi"] for lr in [0.01, 0.03, 0.1] for w in ["size", "equal"]
    ],
    "updates": lambda: [
        multi_seed("normalized directions (mean norm)", normalized_direction("mean")),
        multi_seed("normalized directions (median norm)", normalized_direction("median")),
        multi_seed("dissimilarity weights temp=1", dissimilarity_weights(1.0)),
        multi_seed("dissimilarity weights temp=3", dissimilarity_weights(3.0)),
    ],
}

if __name__ == "__main__":
    for g in (sys.argv[1:] or GROUPS):
        print(f"--- {g} ---"); GROUPS[g]()
