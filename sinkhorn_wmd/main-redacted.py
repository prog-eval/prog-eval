#!/usr/bin/env python

"""
    sinkhorn_wmd/main.py
"""

import os
import sys
import argparse
import numpy as np
from time import time
from scipy import sparse

# --
# Sinkhorn

def sinkhorn_wmd(r, c, vecs, lamb, max_iter):
    """
        r (np.array):          query vector (sparse, but represented as dense)
        c (sparse.csr_matrix): data vectors, in CSR format.  shape is `(dim, num_docs)`
        vecs (np.array):       embedding vectors, from which we'll compute a distance matrix
        lamb (float):          regularization parameter
        max_iter (int):        maximum number of iterations
    """
    
    return scores


# --
# CLI

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inpath', type=str, default='data/cache')
    parser.add_argument('--n-docs', type=int, default=5000)
    parser.add_argument('--query-idx', type=int, default=100)
    parser.add_argument('--lamb', type=float, default=1)
    parser.add_argument('--max_iter', type=int, default=15)
    args = parser.parse_args()
    
    # !! In order to check accuracy, you _must_ use these parameters !!
    assert args.inpath == 'data/cache'
    assert args.n_docs == 5000
    assert args.query_idx == 100
    
    return args


if __name__ == "__main__":
    args = parse_args()
    
    # --
    # IO
    
    # vecs: (num_words, word_embedding_dim) matrix of word embeddings
    # mat:  (num_words, num_docs) sparse term-document matrix in CSR format
    
    vecs = np.load(args.inpath + '-vecs.npy')
    mat  = sparse.load_npz(args.inpath + '-mat.npz')
    
    # --
    # Prep
    
    # Maybe subset docs
    if args.n_docs:
        mat  = mat[:,:args.n_docs]
    
    # --
    # Run
    
    # Get query vector
    r = np.asarray(mat[:,args.query_idx].todense()).squeeze()
    
    t = time()
    scores = sinkhorn_wmd(r, mat, vecs, lamb=args.lamb, max_iter=args.max_iter)
    elapsed = time() - t
    print('elapsed=%f' % elapsed, file=sys.stderr)
    
    # --
    # Write output
    
    os.makedirs('results', exist_ok=True)
    
    np.savetxt('results/scores', scores, fmt='%.8e')
    open('results/elapsed', 'w').write(str(elapsed))