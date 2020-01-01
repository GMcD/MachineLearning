#!/bin/bash

set -e

echo "Running Three training sessions of VehicleNet from a clean slate."

python3 main.py --clearstate --iterations 1
python3 main.py --iterations 2
python3 main.py --iterations 1

tensorboard --logdir=runs