"""Aggregation defenses for the Advanced track. All are factories -> agg(local_models, sizes, prev).
Server knowledge: models + reported sizes only (same rule as the Intermediate track)."""
from adv_common import *

BATCH = 256


def deltas_of(lm, prev):
    return flats_of(lm) - get_flat_params(prev)


def _apply(prev, upd):
    return model_from_flat(prev, get_flat_params(prev) + upd)


# ---- baselines from the notebook --------------------------------------------------------
def median_():
    def factory():
        return lambda lm, sizes, prev: coordinate_median(lm)
    return factory


def notebook_clip(clip_norm=5.0):
    """The notebook's my_defense_fn skeleton: clip each update to L2 <= 5, size-weighted mean."""
    def factory():
        def agg(lm, sizes, prev):
            D = deltas_of(lm, prev)
            D = D * (clip_norm / D.norm(dim=1, keepdim=True).clamp_min(clip_norm))
            return _apply(prev, weighted_mean(D, sizes))
        return agg
    return factory


# ---- standard Byzantine-robust aggregators ----------------------------------------------
def trimmed_mean(k=1):
    """Coordinate-wise trimmed mean (Yin et al. 2018): drop the k largest and k smallest values."""
    def factory():
        def agg(lm, sizes, prev):
            D = deltas_of(lm, prev).sort(dim=0).values
            return _apply(prev, D[k:len(D) - k].mean(dim=0))
        return agg
    return factory


def _krum_scores(D, f):
    dist = torch.cdist(D, D) ** 2
    m = len(D) - f - 2
    return torch.stack([dist[i][torch.arange(len(D)) != i].sort().values[:m].sum() for i in range(len(D))])


def krum(f=1, m=1):
    """Krum / Multi-Krum (Blanchard et al. 2017): keep the m updates with the smallest summed
    distance to their n-f-2 nearest neighbours, average them. Flags the rest."""
    def factory():
        flags = []
        def agg(lm, sizes, prev):
            D = deltas_of(lm, prev)
            keep = _krum_scores(D, f).argsort()[:m].tolist()
            flags.append({i for i in range(len(D)) if i not in keep})
            return _apply(prev, D[keep].mean(dim=0))
        agg.flags = flags
        return agg
    return factory


def geomed(iters=50, eps=1e-8):
    """Geometric median via Weiszfeld (RFA, Pillutla et al. 2022), equal weights."""
    def factory():
        def agg(lm, sizes, prev):
            D = deltas_of(lm, prev); z = D.mean(dim=0)
            for _ in range(iters):
                w = 1.0 / (D - z).norm(dim=1).clamp_min(eps)
                z = (w.unsqueeze(1) * D).sum(0) / w.sum()
            return _apply(prev, z)
        return agg
    return factory


def fednova_equal():
    """Our Intermediate-track method, undefended."""
    def factory():
        def agg(lm, sizes, prev):
            tau = np.ceil(np.array(sizes, float) / BATCH)
            d = deltas_of(lm, prev) / torch.tensor(tau, dtype=torch.float32).unsqueeze(1)
            return _apply(prev, float(tau.mean()) * d.mean(dim=0))
        return active_only(agg)
    return factory


# ---- ours ----------------------------------------------------------------------------------
def fednova_guard(kappa=5.0, clip=3.0, strikes=2, step="mean", tau_cap=25.0):
    """FedNova-Guard. Per-step directions d_i = delta_i / tau_i (FedNova, tau from reported size).
    Detection is on magnitude only, never direction: under non-IID data an honest minority client
    and a stealthy attacker are indistinguishable by direction (see docs), so direction filters
    shoot the honest experts.
      1. m = median per-step norm over non-quarantined clients (honest while < half are malicious)
      2. flag ||d_i|| > kappa * m -> excluded this round; after `strikes` flags, quarantined for good
      3. clip the survivors to clip * m, so an undetected attacker's pull is bounded
      4. equal-weight mean, server step = mean tau of survivors (plain FedNova). step="median" resists
         size-inflation lies but stalls federations dominated by one huge honest bank (see docs)."""
    def factory():
        strike = {}
        flags = []
        def agg(lm, sizes, prev):
            n = len(lm)
            tau = np.ceil(np.maximum(np.array(sizes, float), 1) / BATCH)
            d = deltas_of(lm, prev) / torch.tensor(tau, dtype=torch.float32).unsqueeze(1)
            nrm = d.norm(dim=1)
            pool = [i for i in range(n) if strike.get(i, 0) < strikes and nrm[i] > 0]
            flagged = {i for i in range(n) if strike.get(i, 0) >= strikes}
            if pool:
                m = nrm[pool].median()
                for i in pool:
                    if nrm[i] > kappa * m:
                        flagged.add(i); strike[i] = strike.get(i, 0) + 1
            flags.append(flagged)
            keep = [i for i in pool if i not in flagged]
            if not keep:
                return copy.deepcopy(prev)
            dk = d[keep]
            if clip is not None:
                dk = dk * (clip * m / dk.norm(dim=1, keepdim=True).clamp_min(clip * m))
            if step == "median":
                length = float(np.median(tau[keep]))
            elif step == "capmean":   # mean tau, but no bank may claim > tau_cap x the median bank's steps
                length = float(np.minimum(tau[keep], tau_cap * np.median(tau[keep])).mean())
            else:
                length = float(tau[keep].mean())
            return _apply(prev, length * dk.mean(dim=0))
        agg.flags = flags
        return agg
    return factory


def fednova_oracle(exclude=(1,)):
    """Upper reference: FedNova equal that magically knows the attacker and drops it every round."""
    def factory():
        def agg(lm, sizes, prev):
            keep = [i for i in range(len(lm)) if i not in exclude and sizes[i] > 0]
            tau = np.ceil(np.array([sizes[i] for i in keep], float) / BATCH)
            d = deltas_of([lm[i] for i in keep], prev) / torch.tensor(tau, dtype=torch.float32).unsqueeze(1)
            return _apply(prev, float(tau.mean()) * d.mean(dim=0))
        return agg
    return factory
