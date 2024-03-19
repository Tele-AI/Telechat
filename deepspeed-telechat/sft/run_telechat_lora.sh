#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

OUTPUT=telechat-lora-test
ZERO_STAGE=3
MAX_LEN=4096
NUM_SAMPLES=1000
DATA_OUTPUT_PATH=datas/data_files


if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=3
fi
mkdir -p $OUTPUT

python -u process_data.py \
   --data_path data.json  \
   --tokenizer_path ../../models/12B \
   --data_output_path $DATA_OUTPUT_PATH \
   --max_seq_len $MAX_LEN \
   --num_samples $NUM_SAMPLES \
   --num_workers 10 \
   --process_method multiple \
   --seed 42

deepspeed --master_port 29500 main.py \
   --data_path ${DATA_OUTPUT_PATH}/train_data.pt  \
   --model_name_or_path ../../models/12B \
   --with_loss_mask \
   --per_device_train_batch_size 1 \
   --max_seq_len $MAX_LEN \
   --learning_rate 3e-5 \
   --weight_decay 0.0001 \
   --num_train_epochs 1 \
   --gradient_accumulation_steps 4 \
   --lr_scheduler_type cosine \
   --precision fp16 \
   --warmup_proportion 0.1 \
   --gradient_checkpointing \
   --seed 42 \
   --zero_stage $ZERO_STAGE \
   --save_steps 10 \
   --deepspeed \
   --lora_dim 8 \
   --mark_only_lora_as_trainable \
   --lora_module_name "self_attention." \
   --output_dir $OUTPUT \
   2>&1 | tee $OUTPUT/training.log

