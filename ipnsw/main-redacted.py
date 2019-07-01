#!/usr/bin/env python

"""
    ipnsw/main.py
"""

import os
import sys
import pickle
import argparse
import numpy as np
from time import time

# --
# IP-NSW implementation

def dist(a, b):
    # ... user code here ...
    return distance


def single_search_knn(G0, Gs, data, q, v_entry, ef, n_results):
    """
        NOTE: num_calls_to_dist_function must count the number of times that `dist` is called
    """
    
    # ... user code here ...
    return results, num_calls_to_dist_function


def search_knn(G0, Gs, data, queries, v_entry, ef, n_results):
    """
        We're processing multiple queries one-by-one serially,
        but you can and should try to optimize this (for example,
        by running the queries in parallel)
        
        G0:        graph
        Gs:        list of graphs
        data:      matrix of shape (num_items_in_database, dim)
        v_entry:   id of entrypoint node
        ef:        width of beam search
        n_results: number of items to return
    """
    
    all_results = []
    total_calls_to_dist_function = 0
    for q in queries:
        results, num_calls_to_dist_function = single_search_knn(G0, Gs, data, q, v_entry, ef, n_results)
        all_results.append(results)
        total_calls_to_dist_function += num_calls_to_dist_function
    
    return np.stack(all_results), total_calls_to_dist_function

# --
# CLI

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-queries', type=int, default=512)
    parser.add_argument('--v-entry', type=int, default=82026)
    parser.add_argument('--ef', type=int, default=128)
    parser.add_argument('--n-results', type=int, default=10)
    args = parser.parse_args()
    
    # !! Need these parameters for correctness checking
    assert args.n_queries == 512
    assert args.v_entry == 82026
    assert args.ef == 128
    assert args.n_results == 10
    
    return args


if __name__ == "__main__":
    args = parse_args()
    
    # --
    # IO
    
    # graphs:   Each element of `graphs` is a dictionary whose key is a vertex ID and whose value is a list of neighbor vertex IDs.
    # queries: (10000, 100) matrix of query vectors
    # data:    (1000000, 100) matrix of database vectors
    
    graphs = pickle.load(open('data/music.graphs.pkl', 'rb'))
    G0     = graphs[0]
    Gs     = [graphs[3], graphs[2], graphs[1]]
    
    data = np.fromfile('data/database_music100.bin', dtype=np.float32).reshape(1_000_000, 100)
    
    queries = np.fromfile('data/query_music100.bin', dtype=np.float32).reshape(10_000, 100)
    queries = queries[:args.n_queries]
    
    # --
    # Run
    
    t = time()
    scores, total_calls_to_dist_function = search_knn(
        G0=G0,
        Gs=Gs, 
        data=data,
        queries=queries,
        v_entry=args.v_entry,
        ef=args.ef,
        n_results=args.n_results
    )
    elapsed = time() - t
    
    # --
    # Write output
    
    os.makedirs('results', exist_ok=True)
    
    np.savetxt('results/scores', scores, fmt='%d')
    open('results/counter', 'w').write(str(total_calls_to_dist_function))
    open('results/elapsed', 'w').write(str(elapsed))

