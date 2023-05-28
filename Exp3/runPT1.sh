#!/bin/bash

MODE=PT1
OUTPUT_PATH=./Exp/3/$MODE
mkdir -p $OUTPUT_PATH

# 16 128 768 6*6 target mask rate ((128/16)^2)/((768/16)^2) 0.278
# deepspeed main.py \
# deepspeed --hostfile=host main.py \
deepspeed --hostfile=host main.py \
   --data_sample_input_path /public/home/hydeng/Workspace/yrqUni/unicornEarth/DATA_Demo/Merge/ \
   --data_padmask_input_path /public/home/hydeng/Workspace/yrqUni/unicornEarth/DATA_Demo/PadMask/ \
   --val_rate 0.1 \
   --pretrain_mask_rate 0.5 \
   --data_info /public/home/hydeng/Workspace/yrqUni/unicornEarth/data/DataInfo \
   --target_num_patches 4096 \
   --patch_per_var_side 2 \
   --init_model unicornEarth-base \
   --train_stage PT1 \
   --ckpt_output_dir $OUTPUT_PATH \
   --data_output_path $OUTPUT_PATH \
   --seed 1017 \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 1 \
   --do_eval \
   --learning_rate 5e-4 \
   --weight_decay 0.1 \
   --num_train_epochs 96 \
   --gradient_accumulation_steps 1 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   &> $OUTPUT_PATH/train.log