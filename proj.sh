#!/bin/bash

set -e

# kill -9 $(ps aux | grep tensorboard | grep -v grep | awk '{ print $2 }')
# rm -rf ./runs/*
python3 main.py --iterations 1

tensorboard --logdir=runs
