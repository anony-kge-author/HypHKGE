#!/bin/bash
cd .. 
source set_env.sh
python run.py \
            --dataset FB237 \
            --model HypHKGE \
            --rank 32 \
            --regularizer N3 \
            --reg 0 \
            --optimizer Adagrad \
            --max_epochs 150 \
            --patience 10 \
            --valid 5 \
            --batch_size 500 \
            --neg_sample_size 100 \
            --init_size 0.001 \
            --learning_rate 0.05 \
            --gamma 0.0 \
            --bias learn \
            --dtype double \
            --multi_c
cd examples/
