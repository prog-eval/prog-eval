#!/usr/bin/env python

"""
    utils/prep_ml.py
    
    Data available at: http://files.grouplens.org/datasets/movielens/ml-10m.zip
"""

import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inpath', type=str, default='data/ml-10m.ratings.dat')
    parser.add_argument('--outpath', type=str, default='data/cache')
    parser.add_argument('--seed', type=int, default=789)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    np.random.seed(args.seed)
    
    # --
    # IO
    
    print('loading %s' % args.inpath, file=sys.stderr)
    edges = pd.read_csv(args.inpath, header=None, sep='::', engine='python')[[0, 1]]
    edges.columns = ('userId', 'movieId')
    
    # --
    # Remap IDs to sequential integers
    
    uusers       = set(edges.userId)
    user_lookup  = dict(zip(uusers, range(len(uusers))))
    edges.userId = edges.userId.apply(user_lookup.get)
    
    umovies       = set(edges.movieId)
    movie_lookup  = dict(zip(umovies, range(len(umovies))))
    edges.movieId = edges.movieId.apply(movie_lookup.get)
    edges.movieId += 1 # Add padding character
    
    # --
    # Train/test split
    
    train, valid = train_test_split(edges, train_size=0.8, stratify=edges.userId)
    
    # --
    # Convert to adjacency list + save
    
    train_adjlist = train.groupby('userId').movieId.apply(lambda x: sorted(set(x))).values
    valid_adjlist = valid.groupby('userId').movieId.apply(lambda x: sorted(set(x))).values
    
    print('saving %s' % args.outpath, file=sys.stderr)
    np.save(args.outpath + '_train', train_adjlist)
    np.save(args.outpath + '_valid', valid_adjlist)
