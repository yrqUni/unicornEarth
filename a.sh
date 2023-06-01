#!/bin/bash

NAME=FT
OUTPUT_PATH=./Exp/E-SwinV1-X/$NAME
mkdir -p $OUTPUT_PATH

# 16 128 768 6*6 target mask rate ((128/16)^2)/((768/16)^2) 0.278
# deepspeed main.py \
# deepspeed --hostfile=host main.py \
deepspeed main.py \
   --data_sample_input_path ./DATA_Demo/Merge/ \
   --data_padmask_input_path ./DATA_Demo/PadMask/ \
   --val_rate 0.1 \
   --data_info ./data/DataInfoDemo \
   --stats_path ./data/Stats/ \
   --target_var TCWV \
   --target_num_patches 1024 \
   --patch_per_var_side 32 \
   --model SwinV1 \
   --init_model unicornEarth-SwinV1-base \
   --train_stage FT \
   --ckpt_output_dir $OUTPUT_PATH \
   --data_output_path $OUTPUT_PATH \
   --seed 1017 \
   --per_device_train_batch_size 4 \
   --per_device_eval_batch_size 8 \
   --do_eval \
   --learning_rate 5e-4 \
   --loss_l1_rate 1.0 \
   --loss_ms_ssim_rate 1.0 \
   --weight_decay 0.1 \
   --num_train_epochs 1024 \
   --gradient_accumulation_steps 1 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
#    &> $OUTPUT_PATH/train.log

