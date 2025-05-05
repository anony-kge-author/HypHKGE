cd .. 
source set_env.sh
python run.py \
            --dataset YAGO3-10 \
            --model HypHKGE \
            --rank 32 \
            --regularizer N3 \
            --reg 0.0 \
            --optimizer Adam \
            --max_epochs 350 \
            --patience 15 \
            --valid 5 \
            --batch_size 2000 \
            --neg_sample_size 100 \
            --init_size 0.001 \
            --learning_rate 0.001 \
            --gamma 0.0 \
            --bias learn \
            --dtype double \
            --multi_c
cd examples/
