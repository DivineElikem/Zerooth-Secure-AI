"""Stage 1: reference points and the simplest privacy-safe re-weightings.
Server knowledge: client models + client sizes only."""
from common import *


def static_weights(weight_fn):
    return lambda: (lambda lm, sizes, prev: model_from_flat(lm[0], weighted_mean(flats_of(lm), weight_fn(np.array(sizes, float)))))


if __name__ == "__main__":
    multi_seed("IID  | naive FedAvg (reference)", clients=iid_clients)
    multi_seed("non-IID | naive FedAvg (baseline)")
    multi_seed("non-IID | equal weights", static_weights(lambda n: np.ones_like(n)))
    multi_seed("non-IID | sqrt(size) weights", static_weights(np.sqrt))
    multi_seed("non-IID | log(size) weights", static_weights(np.log))
