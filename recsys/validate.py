#!/usr/bin/env python

"""
    main.py
"""

import sys
import json
import argparse
import numpy as np
from helpers import compute_scores

P_AT_01_THRESHOLD = 0.475

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-path', type=str, default='data/cache')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # --
    # Check correctness
    
    X_valid = np.load('%s_valid.npy' % args.cache_path, allow_pickle=True)
    topk    = np.loadtxt('results/topk')
    
    scores = compute_scores(topk, X_valid)
    
    # --
    # Log
    
    print(json.dumps({
        "status"  : "PASS" if scores[1] >= P_AT_01_THRESHOLD else "FAIL",
        "p_at_01" : scores[1],
        "p_at_05" : scores[5],
        "p_at_10" : scores[10],
    }))
    
    does_pass = "PASS" if scores[1] >= P_AT_01_THRESHOLD else "FAIL"
    open('results/.pass', 'w').write(str(does_pass))
