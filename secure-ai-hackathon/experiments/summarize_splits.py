"""Summarise results/intermediate_splits.jsonl: per split, each method's gain over naive FedAvg."""
import json, re, os
import pandas as pd
rows = []
for line in open(os.path.join(os.path.dirname(__file__), "results", "intermediate_splits.jsonl")):
    r = json.loads(line)
    m = re.match(r"\[alpha=(\S+) split=(\d+)\] (.*)", r["name"])
    rows.append({"alpha": float(m[1]), "split": int(m[2]), "method": m[3], "f1": r["f1_mean"]})
df = pd.DataFrame(rows).drop_duplicates(["alpha", "split", "method"], keep="last")  # later re-runs win
base = df[df.method == "naive FedAvg"].set_index(["alpha", "split"]).f1
df["gain"] = df.apply(lambda r: r.f1 - base[(r.alpha, r.split)], axis=1)
pd.set_option("display.width", 200)
print(df.pivot_table(index=["alpha", "split"], columns="method", values="f1").round(4).to_string())
print("\nmean gain over naive FedAvg by alpha:")
print(df.pivot_table(index="method", columns="alpha", values="gain", aggfunc="mean").round(4).to_string())
print("\nsplits where method beats naive FedAvg (of 9):")
print(df[df.method != "naive FedAvg"].groupby("method").gain.apply(lambda g: f"{(g > 0).sum()}/{len(g)}  worst {g.min():+.4f}").to_string())
