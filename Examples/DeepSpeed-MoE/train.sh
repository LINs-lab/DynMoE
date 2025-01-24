#!/bin/bash

pip install -e ../../DeepSpeed-0.9.5

deepspeed train.py -a dynmoevit_b_16 \
                   --deepspeed \
                   --deepspeed_config ds_config.json \
                   --multiprocessing_distributed \
                   --batch-size 512 \
                   --epochs 300 --seed 42 \
                   --lr 5e-4 --weight-decay 3e-5 \
                   --data path/to/imagenet-1k