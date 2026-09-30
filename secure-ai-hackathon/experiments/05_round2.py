"""Stage 5: can we beat FedNova (equal weights)? Still privacy-safe: models + sizes only.
Every variant is judged on F1 *and* AUC (to catch threshold-shift gains), and any winner must
also beat FedNova (equal) on the 9 alternative splits before we adopt it."""
import sys
from common import *
from importlib import import_module
s2 = import_module("02_privacy_safe")
s3 = import_module("03_combine_and_robustness")

BATCH = 256


def fednova_plus(step="mean", norm_q=0.0, conflict=False, momentum=0.0):
    """FedNova per-step directions with optional extras:
    step      -- server step length: "mean" tau (plain FedNova), "median" tau, or a multiple of mean tau
    norm_q    -- client weights proportional to ||per-step update||**q (0 = equal). Clients the global
                 model fits worst move furthest per step, so q > 0 approximates loss-based fairness
                 (q-FFL) without the clients reporting a loss.
    conflict  -- PCGrad/FedFV-style: remove from each client's direction the component that
                 conflicts (negative dot product) with another client's direction.
    momentum  -- server momentum on the aggregated step (FedAvgM on top of FedNova)."""
    def factory():
        st = {"v": None}
        def agg(lm, sizes, prev):
            tau = np.ceil(np.array(sizes, float) / BATCH)
            d = s2.deltas_of(lm, prev) / torch.tensor(tau, dtype=torch.float32).unsqueeze(1)
            if conflict:
                g = d.clone()
                for i in range(len(d)):
                    for j in np.random.permutation(len(d)):
                        dot = g[i] @ d[j]
                        if j != i and dot < 0:
                            g[i] = g[i] - dot / (d[j] @ d[j]) * d[j]
                d = g
            p = d.norm(dim=1).numpy() ** norm_q
            p = p / p.sum()
            if step == "mean":
                length = tau.mean()
            elif step == "median":
                length = np.median(tau)
            else:
                length = float(step) * tau.mean()
            upd = length * weighted_mean(d, p)
            if momentum > 0:
                st["v"] = upd if st["v"] is None else momentum * st["v"] + upd
                upd = st["v"]
            return model_from_flat(prev, get_flat_params(prev) + upd)
        return active_only(agg)
    return factory


def run_scaffold(client_dfs, agg_fn=None, rounds=8, lr=0.05, weighting="equal", normalize=False,
                 model_fn=lambda: WeakMLP()):
    """SCAFFOLD (Karimireddy et al., 2020), option II control variates. Each client corrects its
    local gradient by (c - c_i) so it follows the *global* direction instead of drifting to its own
    optimum. Clients also upload delta-c_i, which is derived from their update and reveals no more
    than the update itself. normalize=True applies FedNova's per-step normalisation on top."""
    global_model = model_fn()
    client_tensors = [df_to_tensors(df) for df in client_dfs]
    zeros = lambda: [torch.zeros_like(p) for p in global_model.parameters()]
    c, c_i = zeros(), [zeros() for _ in client_tensors]
    history = []
    for _ in range(rounds):
        x = [p.detach().clone() for p in global_model.parameters()]
        deltas, dcs, taus = [], [], []
        for k, (X, y) in enumerate(client_tensors):
            model = copy.deepcopy(global_model)
            opt = torch.optim.SGD(model.parameters(), lr=lr)
            loss_fn = nn.BCEWithLogitsLoss()
            perm = torch.randperm(len(X)); steps = 0
            for i in range(0, len(X), BATCH):
                idx = perm[i:i + BATCH]
                opt.zero_grad()
                loss_fn(model(X[idx]), y[idx]).backward()
                for p, cg, ck in zip(model.parameters(), c, c_i[k]):
                    p.grad += cg - ck
                opt.step(); steps += 1
            y_k = [p.detach() for p in model.parameters()]
            new_ck = [ck - cg + (xp - yp) / (steps * lr) for ck, cg, xp, yp in zip(c_i[k], c, x, y_k)]
            dcs.append([n - o for n, o in zip(new_ck, c_i[k])]); c_i[k] = new_ck
            deltas.append(torch.cat([(yp - xp).view(-1) for yp, xp in zip(y_k, x)])); taus.append(steps)
        n = np.array([len(X) for X, _ in client_tensors], float)
        w = np.ones_like(n) if weighting == "equal" else n
        D = torch.stack(deltas)
        if normalize:
            t = np.array(taus, float)
            upd = t.mean() * weighted_mean(D / torch.tensor(t, dtype=torch.float32).unsqueeze(1), w)
        else:
            upd = weighted_mean(D, w)
        set_flat_params(global_model, get_flat_params(global_model) + upd)
        c = [cg + torch.stack([dc[j] for dc in dcs]).mean(0) for j, cg in enumerate(c)]
        history.append(evaluate(global_model))
    return global_model, history


GROUPS = {
    "reference": lambda: [
        multi_seed("naive FedAvg", None),
        multi_seed("FedNova (equal)  [current pick]", fednova_plus()),
    ],
    "step": lambda: [
        multi_seed("FedNova step=median tau", fednova_plus(step="median")),
        multi_seed("FedNova step=0.5 x mean tau", fednova_plus(step=0.5)),
        multi_seed("FedNova step=1.5 x mean tau", fednova_plus(step=1.5)),
        multi_seed("FedNova step=2.0 x mean tau", fednova_plus(step=2.0)),
    ],
    "momentum": lambda: [
        multi_seed("FedNova + server momentum 0.3", fednova_plus(momentum=0.3)),
        multi_seed("FedNova + server momentum 0.6", fednova_plus(momentum=0.6)),
    ],
    "fairness": lambda: [
        multi_seed("FedNova, weights ~ ||update||^1", fednova_plus(norm_q=1.0)),
        multi_seed("FedNova, weights ~ ||update||^2", fednova_plus(norm_q=2.0)),
        multi_seed("FedNova, weights ~ ||update||^4", fednova_plus(norm_q=4.0)),
    ],
    "conflict": lambda: [
        multi_seed("FedNova + conflict projection", fednova_plus(conflict=True)),
        multi_seed("FedNova + conflict proj + ||u||^2", fednova_plus(conflict=True, norm_q=2.0)),
    ],
    "scaffold": lambda: [
        multi_seed("SCAFFOLD, size weights", runner=run_scaffold, weighting="size"),
        multi_seed("SCAFFOLD, equal weights", runner=run_scaffold, weighting="equal"),
        multi_seed("SCAFFOLD + FedNova norm, equal", runner=run_scaffold, weighting="equal", normalize=True),
    ],
}

GROUPS["step_splits"] = lambda: [
    multi_seed(f"[alpha={a} split={ps}] {name}", f, clients=make_noniid_partition(train_pool, alpha=a, seed=ps),
               seeds=[0, 1, 2], log="intermediate_splits")
    for a in [0.1, 0.3, 1.0] for ps in [1, 2, 3]
    for name, f in [("FedNova step=0.5", fednova_plus(step=0.5)), ("FedNova step=0.75", fednova_plus(step=0.75))]
]
GROUPS["step_075"] = lambda: [multi_seed("FedNova step=0.75 x mean tau", fednova_plus(step=0.75))]

if __name__ == "__main__":
    for g in (sys.argv[1:] or GROUPS):
        print(f"--- {g} ---"); GROUPS[g]()
