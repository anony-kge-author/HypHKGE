#!/bin/bash
cd .. 
source set_env.sh
python run.py \
            --dataset WN18RR \
            --model HypHKGE\
            --rank 32 \
            --regularizer N3 \
            --reg 0 \
            --optimizer Adam \
            --max_epochs 250 \
            --patience 10 \
            --valid 5 \
            --batch_size 1000 \
            --neg_sample_size 100 \
            --init_size 0.001 \
            --learning_rate 0.001 \
            --gamma 0.0 \
            --bias learn \
            --dtype double \
            --double_neg \
            --multi_c
cd examples/
