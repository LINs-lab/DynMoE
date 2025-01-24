
pip install -e ../../DeepSpeed-0.9.5

CUDA_VISIBLE_DEVICES=1 deepspeed train.py -a dynmoevit_b_16 \
                   --deepspeed \
                   --deepspeed_config ds_config.json \
                   --multiprocessing_distributed \
                   --batch-size 128 \
                   --epochs 300 --seed 42 \
                   --lr 5e-4 --weight-decay 3e-5 \
                   --resume /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/chengzhenglin-240108120069/chengzhenglin/project_adaptive_MoE/Train-Imagenet/dynmoevit_b_16_200_66.47_checkpoint.pth.tar \
                   --evaluate --print-freq 50 \
                   --data /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/chengzhenglin-240108120069/chengzhenglin/data_zoo/imagenet-1k