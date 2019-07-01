#!/usr/bin/env python

"""
    lgc/main-redacted.py
"""

import os
import sys
import argparse
import numpy as np
from time import time
from scipy.io import mmread

# --
# Parallel PR-Nibble

def parallel_pr_nibble(seeds, adj, alpha, epsilon):
    """
        Compute PR-Nibble LGC for each seed in seeds
        !! Note that we use a `for` loop to loop over the seeds. During optimization, 
            you can (and should) do something else, such as using parallelism or 
            reformulating the algorithm using matrix multiplications.
    """
    
    pnib_scores = []
    for seed in seeds:
        # ... user code here ...
        pnib_scores.append(pnib_score)
    
    return pnib_scores


# --
# ISTA algorithm

def ista(seeds, adj, alpha, rho, iters):
    """
        Compute ISTA LGC for each seed in seeds
        !! Note that we use a `for` loop to loop over the seeds. During optimization, 
            you can (and should) do something else, such as using parallelism or 
            reformulating the algorithm using matrix multiplications.
    """
    
    ista_scores = []
    for seed in seeds:
        # ... user code here ...
        ista_scores.append(ista_score)
    
    return np.column_stack(ista_scores)

# --
# Run

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-seeds', type=int, default=50)
    parser.add_argument('--alpha', type=float, default=0.15)
    parser.add_argument('--pnib-epsilon', type=float, default=1e-6)
    parser.add_argument('--ista-rho', type=float, default=1e-5)
    parser.add_argument('--ista-iters', type=int, default=50)
    args = parser.parse_args()
    
    # !! In order to check accuracy, you _must_ use these parameters !!
    assert args.num_seeds == 50
    assert args.alpha == 0.15
    assert args.pnib_epsilon == 1e-6
    assert args.ista_rho == 1e-5
    assert args.ista_iters == 50
    
    return args


if __name__ == "__main__":
    args = parse_args()
    
    # --
    # IO
    
    # adj: adjacency matrix of input graph.  Stored on disk in "matrix market" format.
    
    adj = mmread('data/jhu.mtx').tocsr()
    
    # PNIB: Use first `num_seeds` nodes as seeds
    # ISTA: Faster algorithm, so use more seeds to get roughly comparable total runtime
    pnib_seeds = list(range(args.num_seeds))
    ista_seeds = list(range(10 * args.num_seeds))
    
    # --
    # Run Parallel PR-Nibble
    
    t = time()
    pnib_scores = parallel_pr_nibble(pnib_seeds, adj, alpha=args.alpha, epsilon=args.pnib_epsilon)
    assert pnib_scores.shape[0] == adj.shape[0]
    assert pnib_scores.shape[1] == len(pnib_seeds)
    pnib_elapsed = time() - t
    print('parallel_pr_nibble: elapsed = %f' % pnib_elapsed, file=sys.stderr)
    
    # --
    # Run ISTA
    
    t = time()
    ista_scores = ista(ista_seeds, adj, alpha=args.alpha, rho=args.ista_rho, iters=args.ista_iters)
    assert ista_scores.shape[0] == adj.shape[0]
    assert ista_scores.shape[1] == len(ista_seeds)
    ista_elapsed = time() - t
    print('ista: elapsed = %f' % ista_elapsed, file=sys.stderr)
    
    # --
    # Write output
    
    os.makedirs('results', exist_ok=True)
    
    np.savetxt('results/pnib_score.txt', pnib_scores)
    np.savetxt('results/ista_score.txt', ista_scores)
    
    open('results/pnib_elapsed', 'w').write(str(pnib_elapsed))
    open('results/ista_elapsed', 'w').write(str(ista_elapsed))

