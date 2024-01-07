#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
export CUDA_VISIBLE_DEVICES=0,1,2,3
OUTPUT=telechat-lora-test
ZERO_STAGE=1

if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=3
fi
mkdir -p $OUTPUT


deepspeed --master_port 29500 main.py \
   --data_path ../../example_datas/single_turn_example.jsonl \
   --model_name_or_path ../../models/7B \
   --with_loss_mask \
   --data_output_path /tmp/data_files/ \
   --per_device_train_batch_size 1 \
   --max_seq_len 1024 \
   --learning_rate 3e-6 \
   --weight_decay 0.0001 \
   --num_train_epochs 5 \
   --gradient_accumulation_steps 16 \
   --lr_scheduler_type cosine \
   --warmup_proportion 0.1 \
   --seed 1233 \
   --zero_stage $ZERO_STAGE \
   --deepspeed \
   --lora_dim 8 \
   --mark_only_lora_as_trainable \
   --lora_module_name "self_attention." \
   --output_dir $OUTPUT \
   2>&1 | tee $OUTPUT/training.log
