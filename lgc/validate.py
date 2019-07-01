#!/usr/bin/env python

"""
    lgc/main.py
"""

import json
import numpy as np
from scipy.stats import spearmanr

PNIB_THRESH = 0.999
ISTA_THRESH = 0.999

# --
# Helpers

def compute_score(targets, scores):
    assert targets.shape == scores.shape
    n = scores.shape[1]
    return min([spearmanr(scores[:,i], targets[:,i]).correlation for i in range(n)])


if __name__ == "__main__":
    
    # --
    # Check Parallel PR-Nibble correctness
    
    pnib_targets = np.loadtxt('ref/pnib_target.txt.gz')
    pnib_scores  = np.loadtxt('results/pnib_score.txt')
    pnib_score   = compute_score(pnib_targets, pnib_scores)
    
    # --
    # Check ISTA correctness
    
    ista_targets = np.loadtxt('ref/ista_target.txt.gz')
    ista_scores  = np.loadtxt('results/ista_score.txt')
    ista_score   = compute_score(ista_targets, ista_scores)
    
    # --
    # Log
    
    print(json.dumps({
        "pnib_score" : pnib_score,
        "pnib_pass"  : "PASS" if pnib_score > PNIB_THRESH else "FAIL",
        "ista_score" : ista_score,
        "ista_pass"  : "PASS" if ista_score > ISTA_THRESH else "FAIL",
    }))
    
    does_pass = "PASS" if (pnib_score > PNIB_THRESH) and (ista_score > ISTA_THRESH) else "FAIL"
    open('results/.pass', 'w').write(str(does_pass))
