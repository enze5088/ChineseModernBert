#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_GPU=8
PORT_ID=$(expr $RANDOM + 1000)
export OMP_NUM_THREADS=1
torchrun --nproc_per_node $NUM_GPU --master_port $PORT_ID run_mlm.py \
    --model_name_or_path './ModernCBert-Large/' \
    --train_file ./pretrain/CCI3-HQ/ \
    --output_dir ./result/ModernChineseBert-Large/ \
    --learning_rate 1e-4 \
    --num_train_epochs 1 \
    --max_eval_samples 20000 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --save_total_limit 3 \
    --eval_strategy steps \
    --eval_steps 2000 \
    --save_steps 2000 \
    --logging_steps 20 \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --bf16 \
    --gradient_accumulation_steps 1 \
    --dataloader_num_workers 16 \
    --remove_unused_columns False \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    "$@"
