#!/bin/bash

python main.py --action=train --dataset=${1} --split=${2} \
                --num_epochs=100 \
                --num_layers_PG=11 \
                --num_layers_R=10 \
                --num_R=3
