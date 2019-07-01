#!/usr/bin/env python

"""
    graphsage/validate.py
"""

import json
import argparse
import numpy as np
import pandas as pd

MAE_THRESH  = 3.9
CORR_THRESH = 0.72

# --
# Command line interface

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--meta-path',  type=str, default='data/pokec.meta')
    parser.add_argument('--results-path',  type=str, default='results/preds')
    
    return parser.parse_args()


if __name__ == "__main__":
    
    # --
    # CLI
    
    args = parse_args()
    
    # --
    # Load actual values
    
    meta = pd.read_csv(args.meta_path, sep='\t')
    act  = meta[meta.train_mask != 1].target.values.astype(np.float32)
    
    # --
    # Load predictions
    
    preds = np.loadtxt(args.results_path)
    
    # --
    # Check correctness
    
    mae  = np.abs(preds - act).mean()
    corr = np.corrcoef(preds, act)[0, 1]
    
    # --
    # Log
    
    print(json.dumps({
        "mae" : mae,
        "mae_status" : "PASS" if mae <= MAE_THRESH else "FAIL",
        "corr" : corr,
        "corr_status" : "PASS" if corr >= CORR_THRESH else "FAIL",
    }))
    
    does_pass = ("PASS" if (mae <= MAE_THRESH) and (corr >= CORR_THRESH) else "FAIL")
    open('results/.pass', 'w').write(str(does_pass))
