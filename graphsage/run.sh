#!/bin/bash

# graphsage/run.sh

# --
# Run

python main.py

# --
# Check correctness

python validate.py

# A correct implementation will print something like
# {
#     "mae"         : 3.843572539000845,
#     "mae_status"  : "PASS", 
#     "corr"        : 0.7348676892148983, 
#     "corr_status" : "PASS"
# }