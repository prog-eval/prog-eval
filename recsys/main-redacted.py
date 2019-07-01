#!/usr/bin/env python

"""
    main-redacted.py
"""

import os
import sys
import argparse
import numpy as np

from helpers import compute_topk

# --
# User code

def make_model(emb_dim, hidden_dim, bias_offset, dropout, lr):
    # ... your code here ...
    return model


def make_dataloader(X, batch_size, shuffle):
    # ... your code here ...
    return dataloader


def train_one_epoch(model, dataloader):
    # ... your code here ...
    return model


def predict(model, dataloader):
    # ... your code here ...
    return predictions


# --
# Command line interface

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-path', type=str, default='data/cache')
    
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--emb-dim', type=int, default=800)
    parser.add_argument('--hidden-dim', type=int, default=400)
    
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--bias-offset', type=float, default=-10)
    parser.add_argument('--dropout', type=float, default=0.5)
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # --
    # Load data
    
    # X_train: List of lists.  The i'th element of `X_train` is a list of items that user i "liked".
    #          Stored on disk as a numpy file.  You can look at `prep.py` to see how this file was created.
    
    print('loading cache: start', file=sys.stderr)
    X_train = np.load('%s_train.npy' % args.cache_path, allow_pickle=True)
    print('loading cache: done', file=sys.stderr)
    
    n_toks = np.hstack(X_train).max() + 1
    
    # --
    # Model
    
    model = make_model(
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        bias_offset=args.bias_offset,
        dropout=args.dropout,
        lr=args.lr,
    )
    
    # --
    # Train for one epoch
    
    t = time()
    
    model = train_one_epoch(
        model=model,
        dataloader=make_dataloader(X_train, batch_size=args.batch_size, shuffle=True),
    )
    
    preds = predict(
        model=model,
        dataloader=make_dataloader(X_train, batch_size=args.batch_size, shuffle=False)
    )
    
    elapsed = time() - t
    
    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == len(X_train)
    assert preds.shape[1] == n_toks
    
    # --
    # Write results
    
    os.makedirs('results', exist_ok=True)
    
    topk = compute_topk(X_train, preds)
    np.savetxt('results/topk', topk)

