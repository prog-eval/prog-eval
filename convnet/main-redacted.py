#!/usr/bin/env python

"""
    convnet/main.py
"""

import sys
import json
import argparse
import numpy as np
from time import time

# --
# User code
# Note: Depending on how you implement your model, you'll likely have to change the parameters of these
# functions.  They way their shown is just one possble way that the code could be structured.

def make_model(input_channels, output_classes, residual_block_sizes, scale_alpha, optimizer, lr, momentum):
    # ... your code here ...
    return model


def make_train_dataloader(X, y, batch_size, shuffle):
    # ... your code here ...
    return dataloader


def make_test_dataloader(X, batch_size, shuffle):
    # ... your code here ...
    return dataloader


def train_one_epoch(model, dataloader):
    # ... your code here ...
    return model


def predict(model, dataloader):
    # ... your code here ...
    return predictions

# --
# CLI

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action="store_true")
    parser.add_argument('--num-epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--batch-size', type=int, default=128)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    # --
    # IO
    
    # X_train: tensor of shape (number of train observations, number of image channels, image height, image width)
    # X_test:  tensor of shape (number of train observations, number of image channels, image height, image width)
    # y_train: vector of [0, 1] class labels for each train image
    # y_test:  vector of [0, 1] class labels for each test image (don't look at these to make predictions!)
    
    X_train = np.load('data/cifar2/X_train.npy')
    X_test  = np.load('data/cifar2/X_test.npy')
    y_train = np.load('data/cifar2/y_train.npy')
    y_test  = np.load('data/cifar2/y_test.npy')
    
    # --
    # Define model
    
    model = make_model(
        input_channels=3,
        output_classes=2,
        residual_block_sizes=[
            (16, 32),
            (32, 64),
            (64, 128),
        ],
        scale_alpha=0.125,
        optimizer="SGD",
        lr=args.lr,
        momentum=args.momentum,
    )
    
    # --
    # Train
    
    t = time()
    for epoch in range(args.num_epochs):
        
        # Train
        model = train_one_epoch(
            model=model,
            dataloader=make_train_dataloader(X_train, y_train, batch_size=args.batch_size, shuffle=True)
        )
        
        # Evaluate
        preds = predict(
            model=model,
            dataloader=make_test_dataloader(X_test, batch_size=args.batch_size, shuffle=False)
        )
        
        assert isinstance(preds, np.ndarray)
        assert preds.shape[0] == X_test.shape[0]
        
        test_acc = (preds == y_test.squeeze()).mean()
        
        print(json.dumps({
            "epoch"    : int(epoch),
            "test_acc" : test_acc,
            "time"     : time() - t
        }))
        sys.stdout.flush()
        
    elapsed = time() - t
    print('elapsed', elapsed, file=sys.stderr)
    
    # --
    # Save results
    
    os.makedirs('results', exist_ok=True)
    
    np.savetxt('results/preds', preds, fmt='%d')
    open('results/elapsed', 'w').write(str(elapsed))