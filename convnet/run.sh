#!/bin/bash

# convnet/run.sh

# --
# Run

python main.py

# --
# Check correctness

python validate.py

# A correct answer should print something like
# {"test_acc": 0.9645, "status": "PASS"}