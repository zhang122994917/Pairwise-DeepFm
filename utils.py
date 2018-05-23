import numpy as np


def reciprocal_rank(r):
    r = np.asarray(r).nonzero()[0]
    return 1./(r[0]+1) if r.size else 0.

def dcg_at_k(r):
    r = np.asarray(r)
    return np.sum(r/np.log2(np.arange(2,r.size + 2)))

def ndcg_at_k(r):
    dcg_max = dcg_at_k(sorted(r,reverse = True))
    if not dcg_max:
        return 0.
    return dcg_at_k(r) / dcg_max

def precision_at_k(r,k):
    assert k>= 1
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        raise ValueError('Relevance score length < k')
    return np.mean(r)

def average_precision(r):
    r = np.asarray(r) != 0
    out = [precision_at_k(r,k+1) for k in range(r.size) if r[k]]
    if not out:
        return 0.
    return np.mean(out)