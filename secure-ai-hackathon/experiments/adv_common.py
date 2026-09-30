"""Shared helpers for the Advanced-track (poisoning) experiments.

Same privacy rule as the Intermediate track: an aggregator sees only each client's model and its
(reported) dataset size. Attacks are simulated client-side exactly as in the notebook's run_fl;
run_fl_attack() reproduces run_fl bit-for-bit for attack in {"label_flip", "scale"} with
start_round=0 and no size lie, and adds:

  start_round  -- the client is honest before this round ("partway through, one bank turns")
  attack       -- "label_flip" | "scale" (notebook) |
                  "norm_match": label-flip, then rescale the poisoned update to the norm the
                                client's *honest* update would have had (evades norm checks) |
                  "sign_flip":  send -scale_factor x the client's honest update
  size_lie     -- the malicious client reports this dataset size instead of its true one

Aggregators are factories returning agg(local_models, sizes, prev). An aggregator may expose
agg.flags: a list that it appends the set of client ids it rejected/flagged to, once per round.
"""
from common import *

ADV_LOG = "advanced"


def run_fl_attack(client_dfs, agg_fn=None, rounds=8, epochs=1, malicious_clients=(1,),
                  attack="scale", scale_factor=15.0, start_round=0, size_lie=None,
                  model_fn=lambda: WeakMLP()):
    malicious_clients = list(malicious_clients or [])
    global_model = model_fn()
    client_tensors = [df_to_tensors(df) for df in client_dfs]
    history = []
    for r in range(rounds):
        local_models, sizes = [], []
        global_flat = get_flat_params(global_model)
        for cid, (X, y) in enumerate(client_tensors):
            bad = cid in malicious_clients and r >= start_round
            if bad and attack == "sign_flip":
                lm = local_train(global_model, X, y, epochs=epochs)
                set_flat_params(lm, global_flat - scale_factor * (get_flat_params(lm) - global_flat))
            elif bad and attack == "norm_match":
                honest_norm = (get_flat_params(local_train(global_model, X, y, epochs=epochs)) - global_flat).norm()
                lm = local_train(global_model, X, 1 - y, epochs=epochs)
                d = get_flat_params(lm) - global_flat
                set_flat_params(lm, global_flat + d * (honest_norm / d.norm().clamp_min(1e-12)))
            else:
                lm = local_train(global_model, X, 1 - y if bad else y, epochs=epochs)
                if bad and attack == "scale":
                    set_flat_params(lm, global_flat + scale_factor * (get_flat_params(lm) - global_flat))
            local_models.append(lm)
            sizes.append(size_lie if (bad and size_lie is not None) else len(X))
        prev = global_model
        global_model = fedavg(local_models, sizes) if agg_fn is None else agg_fn(local_models, sizes, prev)
        history.append(evaluate(global_model))
    return global_model, history


def detection_stats(flag_log, malicious, start_round, n_clients=N_CLIENTS):
    """Per-(round, client) confusion counts of an aggregator's flag log against the truth."""
    tp = fp = fn = tn = 0
    for r, flagged in enumerate(flag_log):
        for c in range(n_clients):
            truth = c in malicious and r >= start_round
            f = c in flagged
            tp += truth and f; fp += (not truth) and f; fn += truth and not f; tn += (not truth) and not f
    return dict(tp=tp, fp=fp, fn=fn, tn=tn)


def adv_multi_seed(name, agg_factory=None, clients=None, seeds=SEEDS, log=ADV_LOG, tag=None, **atk):
    """Multi-seed runner for one (defense, attack scenario). Logs F1/AUC and detection stats."""
    clients = noniid_clients if clients is None else clients
    runs, aucs, det = [], [], dict(tp=0, fp=0, fn=0, tn=0)
    t = time.time()
    for s in seeds:
        torch.manual_seed(s); np.random.seed(s)
        agg = None if agg_factory is None else agg_factory()
        m, h = run_fl_attack(clients, agg_fn=agg, **atk)
        runs.append(h); aucs.append(test_auc(m))
        if agg is not None and hasattr(agg, "flags"):
            for k, v in detection_stats(agg.flags, atk.get("malicious_clients", (1,)),
                                        atk.get("start_round", 0)).items():
                det[k] += v
    f1 = np.array([h[-1]["f1"] for h in runs])
    has_det = agg_factory is not None and sum(det.values()) > 0
    summary = {
        "name": name, "tag": tag,
        "attack": {k: (list(v) if isinstance(v, tuple) else v) for k, v in atk.items()},
        "seeds": list(seeds),
        "f1_mean": round(float(f1.mean()), 4), "f1_std": round(float(f1.std()), 4),
        "f1_min": round(float(f1.min()), 4),
        "precision_mean": round(float(np.mean([h[-1]["precision"] for h in runs])), 4),
        "recall_mean": round(float(np.mean([h[-1]["recall"] for h in runs])), 4),
        "auc_mean": round(float(np.mean(aucs)), 4),
        "f1_per_seed": f1.tolist(),
        "f1_by_round_mean": np.round(np.mean([[m["f1"] for m in h] for h in runs], axis=0), 4).tolist(),
        "detection": det if has_det else None,
    }
    with open(os.path.join(RESULTS_DIR, f"{log}.jsonl"), "a") as f:
        f.write(json.dumps(summary) + "\n")
    d = ""
    if has_det:
        d = f"  det TP {det['tp']} FN {det['fn']} FP {det['fp']}"
    print(f"{name:34s} {tag or '':22s} F1 {summary['f1_mean']:.4f} ± {summary['f1_std']:.4f} "
          f"(min {summary['f1_min']:.3f})  AUC {summary['auc_mean']:.3f}{d}  ({time.time() - t:.0f}s)", flush=True)
    return summary
