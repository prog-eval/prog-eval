#!/bin/bash

# ipnsw/run.sh

# --
# Run

python main.py

# --
# Check correctness

python validate.py

# A correct implementation should print something like:
# {"recall": 0.9955078125, "p_dist_evals": 0.003385333984375, "status": "PASS"}