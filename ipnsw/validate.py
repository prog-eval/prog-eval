#!/usr/bin/env python

"""
    ipnsw/validate.py
"""

import sys
import json
import argparse
import numpy as np

RECALL_THRESH = 0.99
EVAL_THRESH   = 0.005
DB_SIZE       = int(1e6) # Hardcoded for music100 dataset

# --
# Helpers

def compute_recall(targets, scores):
    recalls = []
    for t, p in zip(targets, scores):
        recalls.append(len(set.intersection(set(t), set(p))) / len(p))
        
    return np.mean(recalls)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--targets', type=str, default='data/correct10_music100.txt')
    parser.add_argument('--scores', type=str, default='results/scores')
    parser.add_argument('--counter', type=str, default='results/counter')
    return parser.parse_args()

if __name__ == "__main__":
    
    args = parse_args()
    
    # --
    # Load output of user code
    
    fn_counter = int(open(args.counter).read())
    
    scores = [[int(xx) for xx in x.split()] for x in open(args.scores).read().splitlines()]
    scores = np.stack(scores)
    assert scores.shape == (512, 10)
    
    # --
    # Load targets
    
    targets = [[int(xx) for xx in x.split()] for x in open(args.targets).read().splitlines()]
    targets = np.stack(targets)
    targets = targets[:scores.shape[0]]
    
    # --
    # Check correctness
    
    assert targets.shape[0] == scores.shape[0]
    assert targets.shape[1] == scores.shape[1]
    
    recall = compute_recall(targets, scores)
    p_dist_evals = fn_counter / (scores.shape[0] * DB_SIZE)
    
    # --
    # Log
    
    print(json.dumps({
        'recall'       : recall,
        'p_dist_evals' : p_dist_evals,
        'status'       : 'PASS' if (recall > RECALL_THRESH) and (p_dist_evals < EVAL_THRESH) else 'FAIL'
    }), file=sys.stderr)
    
    does_pass = 'PASS' if ((recall > RECALL_THRESH) and (p_dist_evals < EVAL_THRESH)) else 'FAIL'
    open('results/.pass', 'w').write(str(does_pass))
