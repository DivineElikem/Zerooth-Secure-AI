"""Stage 4: is the F1 gain real learning or just a threshold shift toward 'attack'?
Threshold-free metrics, per-family recall on KDDTest+, and the notebook's reserved
in-distribution holdout (15k KDDTrain+ rows no client ever saw)."""
from common import *
from importlib import import_module
from sklearn.metrics import roc_auc_score, precision_recall_curve
from harness import _holdout_idx
s2 = import_module("02_privacy_safe")
s3 = import_module("03_combine_and_robustness")

holdout_df = train_df.iloc[_holdout_idx]
X_ho = torch.tensor(holdout_df[FEATURE_COLS].values, dtype=torch.float32)
y_ho = holdout_df["binary_label"].values
test_fam = test_df["label"].map(lambda x: FAMILY_MAP.get(x, "r2l_u2r_other")).values


def scores(model, X):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(X)).numpy()


def diag(model):
    st, sh = scores(model, X_test), scores(model, X_ho)
    yt = y_test.numpy()
    p, r, _ = precision_recall_curve(yt, st)
    pred = st > 0.5
    out = {"test_auc": roc_auc_score(yt, st),
           "test_bestF1": float(np.max(2 * p * r / np.clip(p + r, 1e-9, None))),
           "holdout_auc": roc_auc_score(y_ho, sh),
           "holdout_f1": f1_score(y_ho, sh > 0.5),
           "test_pred_attack_rate": pred.mean(),
           "test_FPR": pred[yt == 0].mean()}
    for f in ["dos", "probe", "r2l_u2r_other"]:
        out[f"recall_{f}"] = pred[test_fam == f].mean()
    return out


def weight_trace(temp=5.0):
    """Record the FedNova+dissimilarity weights each round (for the writeup)."""
    trace = []
    inner = s3.fednova_dissim(temp)()
    def agg(lm, sizes, prev):
        d = s2.deltas_of(lm, prev) / torch.tensor(np.ceil(np.array(sizes) / s2.BATCH), dtype=torch.float32).unsqueeze(1)
        u = d / d.norm(dim=1, keepdim=True)
        mc = (((u @ u.T).sum(1) - 1) / (len(lm) - 1)).numpy()
        w = np.exp(-temp * mc); trace.append((w / w.sum()).round(3).tolist())
        return inner(lm, sizes, prev)
    return agg, trace


if __name__ == "__main__":
    methods = {"naive FedAvg": None, "equal weights": s2.EQUAL, "FedNova (equal)": s2.fednova("equal"),
               "FedNova + dissim T=3": s3.fednova_dissim(3.0), "FedNova + dissim T=5": s3.fednova_dissim(5.0)}
    rows = {}
    for name, fac in methods.items():
        ds = []
        for s in SEEDS:
            torch.manual_seed(s); np.random.seed(s)
            m, _ = run_fl_ext(noniid_clients, agg_fn=None if fac is None else fac())
            ds.append(diag(m))
        rows[name] = {k: round(float(np.mean([d[k] for d in ds])), 4) for k in ds[0]}
    df = pd.DataFrame(rows).T
    pd.set_option("display.width", 250)
    print(df.to_string())
    df.to_csv(os.path.join(RESULTS_DIR, "intermediate_diagnostics.csv"))

    torch.manual_seed(0); np.random.seed(0)
    agg, trace = weight_trace(5.0)
    run_fl_ext(noniid_clients, agg_fn=agg)
    print("\nFedNova+dissim T=5 weights per round (banks 0..4), seed 0:")
    for r, w in enumerate(trace, 1):
        print(f"  round {r}: {w}")
    print("  size-based FedAvg weights:", (np.array([len(c) for c in noniid_clients]) / len(train_pool)).round(3).tolist())
