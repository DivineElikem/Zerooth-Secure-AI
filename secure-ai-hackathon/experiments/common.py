"""Shared helpers for the intermediate-track experiments.

Privacy rule for every aggregation tried here: the server only sees what the
notebook's harness already gives it -- each client's trained model and its
dataset size. No label or attack-family statistics leave a client.
"""
import json, time
from harness import *
from sklearn.metrics import roc_auc_score

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]


def flats_of(models):
    return torch.stack([get_flat_params(m) for m in models])


def model_from_flat(template, flat):
    out = copy.deepcopy(template)
    set_flat_params(out, flat)
    return out


def weighted_mean(flats, weights):
    w = torch.tensor(np.asarray(weights, dtype=float) / np.sum(weights), dtype=torch.float32)
    return (flats * w.unsqueeze(1)).sum(dim=0)


def local_train_ext(model, X, y, epochs=1, lr=0.05, batch_size=256, prox_mu=0.0, max_steps=None):
    """local_train from the notebook, plus an optional FedProx proximal term
    (prox_mu/2 * ||w - w_global||^2) and an optional cap on SGD steps per epoch.
    With prox_mu=0 and max_steps=None it consumes the RNG identically."""
    global_params = [p.detach().clone() for p in model.parameters()]
    model = copy.deepcopy(model)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    n = len(X)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for step, i in enumerate(range(0, n, batch_size)):
            if max_steps is not None and step >= max_steps:
                break
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(X[idx]), y[idx])
            if prox_mu > 0:
                loss = loss + prox_mu / 2 * sum(((p - g) ** 2).sum()
                                                for p, g in zip(model.parameters(), global_params))
            loss.backward()
            opt.step()
    return model


def run_fl_ext(client_dfs, agg_fn=None, rounds=8, epochs=1, lr=0.05, prox_mu=0.0,
               max_steps=None, model_fn=lambda: WeakMLP()):
    """Honest-clients version of the notebook's run_fl with extra client-side knobs.
    agg_fn=None means plain size-weighted FedAvg."""
    global_model = model_fn()
    client_tensors = [df_to_tensors(df) for df in client_dfs]
    history = []
    for _ in range(rounds):
        local_models, sizes = [], []
        for X, y in client_tensors:
            local_models.append(local_train_ext(global_model, X, y, epochs=epochs, lr=lr,
                                                prox_mu=prox_mu, max_steps=max_steps))
            sizes.append(len(X))
        prev = global_model
        global_model = fedavg(local_models, sizes) if agg_fn is None else agg_fn(local_models, sizes, prev)
        history.append(evaluate(global_model))
    return global_model, history


def test_auc(model):
    model.eval()
    with torch.no_grad():
        return float(roc_auc_score(y_test.numpy(), torch.sigmoid(model(X_test)).numpy()))


def multi_seed(name, agg_factory=None, clients=None, seeds=SEEDS, log="intermediate", runner=None, **run_kw):
    """Run one configuration over several training seeds (the data split stays fixed,
    SEED=42 as in the notebook). agg_factory() must return a fresh agg_fn per seed so
    stateful aggregators (momentum, Adam) don't leak state between runs."""
    clients = noniid_clients if clients is None else clients
    runner = run_fl_ext if runner is None else runner
    runs, aucs = [], []
    t = time.time()
    for s in seeds:
        torch.manual_seed(s); np.random.seed(s)
        m, h = runner(clients, agg_fn=None if agg_factory is None else agg_factory(), **run_kw)
        runs.append(h); aucs.append(test_auc(m))
    final = {k: np.array([h[-1][k] for h in runs]) for k in ["f1", "precision", "recall"]}
    summary = {
        "name": name, "config": {k: v for k, v in run_kw.items()}, "seeds": list(seeds),
        "f1_mean": round(float(final["f1"].mean()), 4), "f1_std": round(float(final["f1"].std()), 4),
        "precision_mean": round(float(final["precision"].mean()), 4),
        "recall_mean": round(float(final["recall"].mean()), 4),
        "auc_mean": round(float(np.mean(aucs)), 4),
        "f1_per_seed": final["f1"].tolist(),
        "f1_by_round_mean": np.round(np.mean([[m["f1"] for m in h] for h in runs], axis=0), 4).tolist(),
    }
    with open(os.path.join(RESULTS_DIR, f"{log}.jsonl"), "a") as f:
        f.write(json.dumps(summary) + "\n")
    print(f"{name:40s} F1 {summary['f1_mean']:.4f} ± {summary['f1_std']:.4f}   "
          f"P {summary['precision_mean']:.3f}  R {summary['recall_mean']:.3f}  AUC {summary['auc_mean']:.4f}   ({time.time() - t:.0f}s)")
    return summary


def active_only(agg):
    """Drop clients that contributed nothing this round (no data, or an all-zero update)
    before aggregating. Without this, an empty client makes FedNova divide by tau=0 and
    cosine-based weights divide by a zero norm -> NaN global model."""
    def wrapped(local_models, sizes, prev):
        prev_flat = get_flat_params(prev)
        keep = [i for i, (m, n) in enumerate(zip(local_models, sizes))
                if n > 0 and (get_flat_params(m) - prev_flat).norm() > 0]
        if not keep:
            return copy.deepcopy(prev)
        return agg([local_models[i] for i in keep], [sizes[i] for i in keep], prev)
    return wrapped
