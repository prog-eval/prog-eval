#!/bin/bash

# lgc/run.sh

# --
# Run

python main.py

# --
# Check correctness

python validate.py

# A correct implementation should print something like
# {
#     "pnib_score": 0.9999999999999999, "pnib_pass": "PASS", 
#     "ista_score": 0.9999999999999998, "ista_pass": "PASS"
# }
# Remember that a correct implementation of this task must pass _both_ PR-Nibble and ISTA