#!/usr/bin/env python

"""
    convnet/prep.py
    
    Grabs the first two classes of CIFAR10 and saves them as numpy arrays
    
    Since we don't assume everyone has access to GPUs, we do this to create
    a dataset that can be trained in a reasonable amount of time on a CPU.
"""

import os
import sys
import numpy as np

import torch
from torchvision import transforms, datasets

if __name__ == "__main__":
    
    # --
    # Load data
    
    print('prep.py: dowloading cifar10', file=sys.stderr)
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
    testset  = datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())
    
    X_train, y_train = zip(*[(x, y) for x, y in trainset if y <= 1])
    X_test, y_test   = zip(*[(x, y) for x, y in testset if y <= 1])
    
    X_train = np.array(torch.stack(X_train)).astype(np.float32)
    X_test  = np.array(torch.stack(X_test)).astype(np.float32)
    y_train = np.array(y_train).astype(np.int64)
    y_test  = np.array(y_test).astype(np.int64)
    
    # --
    # Scale and center data
    
    X_mean = X_train.transpose(1, 0, 2, 3).reshape(3, -1).mean(axis=-1).reshape(1, 3, 1, 1)
    X_std = X_train.transpose(1, 0, 2, 3).reshape(3, -1).std(axis=-1).reshape(1, 3, 1, 1)
    
    X_train = (X_train - X_mean) / X_std
    X_test  = (X_test - X_mean) / X_std
    
    # --
    # Save to file
    
    os.makedirs('data/cifar2', exist_ok=True)
    
    print('prep.py: saving to data/cifar2', file=sys.stderr)
    np.save('data/cifar2/X_train.npy', X_train)
    np.save('data/cifar2/X_test.npy', X_test)
    
    np.save('data/cifar2/y_train.npy', y_train)
    np.save('data/cifar2/y_test.npy', y_test)
