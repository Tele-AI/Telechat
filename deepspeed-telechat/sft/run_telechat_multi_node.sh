#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
OUTPUT=telechat-multi-node-test
ZERO_STAGE=3
HOST=my_hostfile

if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=3
fi
mkdir -p $OUTPUT


deepspeed --master_port 29500 --hostfile=$HOST main.py \
   --data_path ../../example_datas/single_turn_example.jsonl  \
   --model_name_or_path ../../models/7B \
   --with_loss_mask \
   --data_output_path /tmp/data_files/ \
   --per_device_train_batch_size 1 \
   --max_seq_len 2048 \
   --learning_rate 2e-5 \
   --weight_decay 0.0001 \
   --num_train_epochs 1 \
   --gradient_accumulation_steps 1 \
   --lr_scheduler_type cosine \
   --gradient_checkpointing \
   --warmup_proportion 0.1 \
   --seed 1233 \
   --zero_stage $ZERO_STAGE \
   --deepspeed \
   --output_dir $OUTPUT \
   2>&1 | tee $OUTPUT/training.log
