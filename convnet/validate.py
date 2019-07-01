#!/usr/bin/env python

"""
    convnet/validate.py
"""

import sys
import json
import numpy as np

ACC_THRESHOLD = 0.955

if __name__ == '__main__':
    y_test = np.load('data/cifar2/y_test.npy')
    y_pred = np.array([int(xx) for xx in open('results/preds').read().splitlines()])
    
    test_acc = (y_test == y_pred).mean()
    
    # --
    # Log
    
    print(json.dumps({
        "test_acc" : float(test_acc),
        "status"   : "PASS" if test_acc >= ACC_THRESHOLD else "FAIL",
    }))
    
    does_pass = "PASS" if test_acc >= ACC_THRESHOLD else "FAIL"
    open('results/.pass', 'w').write(str(does_pass))
