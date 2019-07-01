#!/usr/bin/env python

"""
    recsys/helpers.py
"""

import numpy as np

def _overlap(x, y):
    return len(set(x).intersection(y))

def compute_topk(X_train, preds, ks=[1, 5, 10]):
    max_k = max(ks)
    
    # --
    # Filter training samples
    
    # !! The model will tend to predict samples that are in the training data
    # and so (by construction) not in the validation data.  We don't want to
    # count these as incorrect though, so we filter them from the predictions
    low_score = preds.min() - 1
    for i, xx in enumerate(X_train):
        preds[i][xx] = low_score
    
    # --
    # Get top-k predictions
    
    # identical to `np.argsort(-pred, axis=-1)[:,:k]`, but should be faster
    topk = np.argpartition(-preds, kth=max_k, axis=-1)[:,:max_k]
    topk = np.vstack([r[np.argsort(-p[r])] for r,p in zip(topk, preds)])
    
    return topk


def compute_scores(topk, X_valid, ks=[1, 5, 10]):
    # --
    # Compute precision-at-k for each value of k
    
    precision_at_k = {}
    for k in ks:
        ps = [_overlap(X_valid[i], topk[i][:k]) for i in range(len(X_valid))]
        precision_at_k[k] = np.mean(ps) / k
    
    return precision_at_k