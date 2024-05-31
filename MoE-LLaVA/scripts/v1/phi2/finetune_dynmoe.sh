#!/bin/bash

moe_mode="sparse"

num_experts=4
top_k_experts=-1
max_expert_num=${num_experts}

use_residual=False
router_aux_loss_coef=0.01

JSON_FOLDER="ft_json"
IMAGE_FOLDER="train_image_video"

# Before running this script, make sure you have installed MoE-LLaVA and our modified Deepspeed

cd "path/to/MoE-LLaVA"

pip install "path/to/Deepspeed"

HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 deepspeed moellava/train/train_mem.py \
    --moe_enable True --num_experts ${num_experts} --max_expert_num ${max_expert_num} --top_k_experts ${top_k_experts} --capacity_factor 1.5 \
    --moe_mode ${moe_mode} --use_residual ${use_residual} --router_aux_loss_coef ${router_aux_loss_coef} \
    --train_modules fc1 fc2 wg \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path ./checkpoints/MoE-LLaVA-Phi2-Stage2 \
    --version phi \
    --data_path ${JSON_FOLDER}/llava_image_tune_.json ${JSON_FOLDER}/nlp_tune.json \
    --image_folder ${IMAGE_FOLDER} \
    --image_tower openai/clip-vit-large-patch14-336 \
    --image_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/llavaphi-2.7b-finetune-dynmoe \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 5000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --cache_dir "./cache_dir"
