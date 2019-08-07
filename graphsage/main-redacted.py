#!/usr/bin/env python

"""
    graphsage/main-redacted.py
    
    Simple skeleton for the implementation, showing how data should be read, 
    and how the evaluation metrics are computed.
    
    
"""

import sys
import argparse
import numpy as np
import pandas as pd
from scipy.io import mmread

# --
# Helpers

def compute_scores(act, preds):
    return {
        "mae"  : np.abs(preds - act).mean(),
        "corr" : np.corrcoef(preds, act)[0, 1],
    }

# --
# User code
# Note: Depending on how you implement your model, you'll likely have to change the parameters of these
# functions.  They way their shown is just one possble way that the code could be structured.

def make_model(adj, feats, n_neibs, hidden_dim, lr, momentum, optimizer, nesterov)
    # ... your code here ...
    return model


def make_train_dataloader(idx, target, batch_size, shuffle):
    # ... your code here ...
    return dataloader


def make_valid_dataloader(idx, batch_size, shuffle):
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
    
    parser.add_argument('--graph-path', type=str, default='data/pokec.mtx')
    parser.add_argument('--meta-path',  type=str, default='data/pokec.meta')
    parser.add_argument('--feat-path',  type=str, default='data/pokec.feat.npy')
    
    parser.add_argument('--n-epochs',     type=int, default=3)
    parser.add_argument('--batch-size',   type=int, default=256)
    parser.add_argument('--n-neibs',      type=int, default=12)
    parser.add_argument('--hidden-dim',   type=int, default=64)
    parser.add_argument('--lr',           type=float, default=0.01)
    parser.add_argument('--momentum',     type=float, default=0.9)
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # --
    # IO
    
    # meta: TSV with columns (node_id, target, train_mask).
    #   - `target` is the value that we're trying to predict
    #   - `train_mask` indicates whether each node is in the train or test data split
    #
    # adj: adjacency matrix of the graph.  This is stored on disk in "matrix market" format.
    # feats: feature matrix (one row per node)
    
    meta  = pd.read_csv(args.meta_path, sep='\t')
    adj   = mmread(args.graph_path).tocsr()
    feats = np.load(args.feat_path)
    
    train_meta = meta[meta.train_mask == 1]
    valid_meta = meta[meta.train_mask != 1]

    train_idx    = train_meta.node_id.values.astype(np.int64)
    train_target = train_meta.target.values.astype(np.float32)

    valid_idx    = valid_meta.node_id.values.astype(np.int64)
    valid_target = valid_meta.target.values.astype(np.float32)

    # --
    # Make model

    model = make_model(adj, feats, args.n_neibs, args.hidden_dim, args.lr, args.momentum, optimizer='sgd', nesterov=True)

    # --
    # Make dataloaders

    t = time()

    train_dataloader = make_train_dataloader(train_idx, train_target, args.batch_size, shuffle=True)
    valid_dataloader = make_valid_dataloader(valid_idx, args.batch_size, shuffle=False)

    for epoch in range(args.n_epochs):
        model = train_one_epoch(model, train_dataloader)
        preds = predict(model, valid_dataloader)
        
        scores = compute_scores(valid_target, preds)
        scores.update({"epoch" : epoch})
        print(scores)

    elapsed = time() - t
    print("elapsed", elapsed, file=sys.stderr)

    # --
    # Save results

    os.makedirs('results', exist_ok=True)

    open('results/elapsed', 'w').write(str(elapsed))
    np.savetxt('results/preds', preds, fmt='%.6e')
