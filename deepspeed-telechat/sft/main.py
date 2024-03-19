#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import math
import sys

import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F

from transformers import (
    SchedulerType,
    default_data_collator,
    get_scheduler,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
)

# from .transformers import AutoModelForCausalLM
# import torch.distributed as dist

import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from transformers.deepspeed import HfDeepSpeedConfig
import json

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils.data.data_utils import create_prompt_dataset
from utils.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, get_optimizer_grouped_parameters, save_zero_three_model
from utils.ds_utils import get_train_ds_config
from utils.module.lora import convert_linear_layer_to_lora, convert_lora_to_linear_layer, only_optimize_lora_parameters, recover_lora, mark_only_lora_as_trainable, make_model_gradient_checkpointing_compatible


def parse_args():
    parser = argparse.ArgumentParser(
        description=
        "Finetune a transformers model on a causal language modeling task")
    parser.add_argument('--data_path',
                        type=str,
                        required=True,
                        help='Path to the training dataset.')
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=512,
        help="The maximum sequence length.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help=
        "Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay",
                        type=float,
                        default=0.,
                        help="Weight decay to use.")
    parser.add_argument("--num_train_epochs",
                        type=int,
                        default=1,
                        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help=
        "Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",
        help="The scheduler type to use.",
        choices=[
            "linear", "cosine", "cosine_with_restarts", "polynomial",
            "constant", "constant_with_warmup"
        ],
    )
    parser.add_argument(
        "--warmup_proportion",
        type=float,
        default=0.1,
        help="Proportion of steps for the warmup in the lr scheduler.")
    parser.add_argument("--output_dir",
                        type=str,
                        default=None,
                        help="Where to store the model.")
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="A seed for reproducible training.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--mark_only_lora_as_trainable",
                        action='store_true',
                        help="mark only lora as trainable")
    parser.add_argument('--gradient_checkpointing',
                        action='store_true',
                        help='Enable HF gradient checkpointing for model.')
    parser.add_argument('--with_loss_mask',
                        action='store_true',
                        help='Whether use loss mask in training phrase')
    parser.add_argument("--user_token",
                        type=str,
                        default="<_user>",
                        help="user token")
    parser.add_argument("--bot_token",
                        type=str,
                        default="<_bot>",
                        help="bot token")
    parser.add_argument("--end_token",
                        type=str,
                        default="<_end>",
                        help="end token")
    parser.add_argument('--disable_dropout',
                        action='store_true',
                        help='Disable the dropout of the model.')
    parser.add_argument("--save_steps",
                        type=int,
                        help="Save model steps")
    # deepspeed features
    parser.add_argument('--offload',
                        action='store_true',
                        help='Enable ZeRO Offload techniques.')
    parser.add_argument(
        '--zero_stage',
        type=int,
        default=0,
        help='ZeRO optimization stage for Actor model (and clones).')
    ## LoRA for efficient training setting
    parser.add_argument("--lora_dim",
                        type=int,
                        default=0,
                        help="If > 0, use LoRA for efficient training.")
    parser.add_argument("--lora_scaling",
                        type=int,
                        default=1,
                        help="use for scaling LoRA matrix.")
    parser.add_argument("--lora_module_name",
                        type=str,
                        default="decoder.layers.",
                        help="The scope of LoRA.")
    parser.add_argument('--only_optimize_lora',
                        action='store_true',
                        help='Only optimize the LoRA parameters.')
    parser.add_argument('--precision',
                        choices=['fp16', 'bf16'],
                        required=True,
                        help='Choose the mixed precision type while training')
    parser.add_argument("--gradient_checkpointing_use_reentrant", action="store_true")
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    # Validate settings
    if args.gradient_checkpointing and args.lora_dim > 0:
        assert (
            not args.only_optimize_lora
        ), "--gradient_checkpointing and --only_optimize_lora cannot be enabled at the same time."

    return args

def load_telechat_tokenizer(model_name_or_path, fast_tokenizer=True):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,
                                              fast_tokenizer=fast_tokenizer,
                                              padding_side="left",
                                              trust_remote_code=True)
    return tokenizer

def create_hf_telechat(model_name_or_path,
                       precision,
                       ds_config=None,
                       disable_dropout=False):
    model_config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    if disable_dropout:
        model_config.dropout = 0.0
    # Note: dschf is defined in function scope to avoid global effects
    # https://huggingface.co/docs/transformers/main_classes/deepspeed#nontrainer-deepspeed-integration
    if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
        dschf = HfDeepSpeedConfig(ds_config)
    else:
        dschf = None
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                 trust_remote_code=True,
                                                 config=model_config,
                                                 torch_dtype=torch.float16 if precision == "fp16" else torch.bfloat16)
    return model

def masked_cross_entropy_loss(logits, labels, loss_mask):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_loss_mask = loss_mask[..., 1:].contiguous()
    shift_logits = F.log_softmax(shift_logits, dim=-1)
    loss = -torch.gather(shift_logits, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
    shift_loss_mask = shift_loss_mask.view(-1)
    expected_number_of_tokens = shift_loss_mask.sum()
    loss = torch.sum(loss.view(-1) * shift_loss_mask) / expected_number_of_tokens
    return loss


def loss_fn(logits, labels, user_token_id, bot_token_id, end_token_id):
    # unpack
    loss_mask = torch.zeros(labels.size(), device=labels.device)
    indices_user = torch.where(labels == user_token_id)[1].tolist()
    indices_bot = torch.where(labels == bot_token_id)[1].tolist()
    indices_end = torch.where(labels == end_token_id)[1].tolist()

    assert len(indices_user) != 0
    assert len(indices_user) == len(indices_bot) == len(indices_end)

    for i in range(len(indices_bot)):
        bot_idx = indices_bot[i]
        end_idx = indices_end[i]
        user_idx = indices_user[i]
        loss_mask[0][bot_idx:end_idx + 1] = 1
        loss_mask[0][user_idx] = 1
    loss = masked_cross_entropy_loss(logits, labels, loss_mask)
    return loss

def main():
    args = parse_args()

    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        # torch.distributed.init_process_group(backend='nccl')
        deepspeed.init_distributed()

    args.global_rank = torch.distributed.get_rank()

    ds_config = get_train_ds_config(offload=args.offload,
                                    stage=args.zero_stage,
                                    precision=args.precision)
    ds_config[
        'train_micro_batch_size_per_gpu'] = args.per_device_train_batch_size
    ds_config[
        'train_batch_size'] = args.per_device_train_batch_size * torch.distributed.get_world_size(
        ) * args.gradient_accumulation_steps
    loss_update_steps = args.per_device_train_batch_size * args.gradient_accumulation_steps

    # If passed along, set the training seed now.
    set_random_seed(args.seed)

    torch.distributed.barrier()

    tokenizer = load_telechat_tokenizer(args.model_name_or_path, fast_tokenizer=True)
    args.user_token_id = tokenizer.convert_tokens_to_ids(args.user_token)
    args.bot_token_id = tokenizer.convert_tokens_to_ids(args.bot_token)
    args.end_token_id = tokenizer.convert_tokens_to_ids(args.end_token)


    model = create_hf_telechat(args.model_name_or_path,
                               args.precision,
                               ds_config,
                               disable_dropout=args.disable_dropout)

    if args.lora_dim > 0:
        model = convert_linear_layer_to_lora(model, args.lora_module_name, args.lora_scaling,
                                             args.lora_dim)
        if args.mark_only_lora_as_trainable:
            mark_only_lora_as_trainable(model, 'lora_only')
            make_model_gradient_checkpointing_compatible(model)
        if args.only_optimize_lora:
            model = only_optimize_lora_parameters(model)

    # Prepare the data
    print(f"train_fname:{args.data_path}")
    assert os.path.exists(args.data_path), "Please process data first!"
    torch.distributed.barrier()
    train_dataset = torch.load(args.data_path)

    # DataLoaders creation:
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  collate_fn=default_data_collator,
                                  sampler=train_sampler,
                                  batch_size=args.per_device_train_batch_size)

    # Split weights in two groups, one with weight decay and the other not.
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(
        model, args.weight_decay)

    AdamOptimizer = DeepSpeedCPUAdam if args.offload else FusedAdam
    optimizer = AdamOptimizer(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              betas=(0.9, 0.95),
                              eps=1e-5)

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    num_warmup_steps = int(args.warmup_proportion * args.num_train_epochs * num_update_steps_per_epoch)
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=args.num_train_epochs * num_update_steps_per_epoch,
    )

    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        config=ds_config,
        lr_scheduler=lr_scheduler,
        dist_init_required=True)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        # model.gradient_checkpointing_enable(
        #     gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
        # )

    # Train!
    print_rank_0("***** Running training *****", args.global_rank)
    global_step = 0
    cur_batch_loss = 0.0
    for epoch in range(args.num_train_epochs):
        print_rank_0(
            f"Beginning of Epoch {epoch+1}/{args.num_train_epochs}, Total Micro Batches {len(train_dataloader)}",
            args.global_rank)
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            batch = to_device(batch, device)
            if args.with_loss_mask:
                outputs = model(batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False)
                logits = outputs.logits
                loss = loss_fn(logits, batch["labels"], args.user_token_id, args.bot_token_id, args.end_token_id)
            else:
                outputs = model(**batch, use_cache=False)
                loss = outputs.loss
            model.backward(loss)
            model.step()
            torch.distributed.reduce(loss, 0)
            total_loss += loss
            if (step + 1) % loss_update_steps == 0:
                cur_batch_loss = total_loss / (loss_update_steps * torch.distributed.get_world_size())
                print_rank_0(f"epoch:{epoch+1}, global_step:{global_step+1}, step:{step+1}  cur_batch_loss: {cur_batch_loss}", args.global_rank)
                global_step += 1
                total_loss = 0.0
            if global_step > 0 and global_step % args.save_steps == 0:
                if args.output_dir is not None:
                    print_rank_0(f'saving step {global_step} model ...', args.global_rank)
                    if args.lora_dim > 0:
                        model = convert_lora_to_linear_layer(model)
                        print_rank_0('convert lora to linear layer successfully!', args.global_rank)

                    if args.zero_stage < 3 and args.global_rank <= 0:
                        save_hf_format(model, tokenizer, args, f"global_step_{global_step}_loss_{cur_batch_loss:.4f}")

                    if args.zero_stage == 3:
                    # For zero stage 3, each gpu only has a part of the model, so we need a special save function
                        save_zero_three_model(model, tokenizer, args, f"global_step_{global_step}")
                    print_rank_0('save successfully!', args.global_rank)
                    if args.lora_dim > 0:
                        print_rank_0('recovering lora...', args.global_rank)
                        model = recover_lora(model)
                        print_rank_0('recover successfully!', args.global_rank)
        model.tput_timer.update_epoch_count()


    if args.output_dir is not None:
        print_rank_0('saving the final model ...', args.global_rank)
        if args.lora_dim > 0:
            model = convert_lora_to_linear_layer(model)
            print_rank_0('convert lora to linear layer successfully!', args.global_rank)

        if args.zero_stage < 3 and args.global_rank == 0:
            save_hf_format(model, tokenizer, args)

        if args.zero_stage == 3:
            # For zero stage 3, each gpu only has a part of the model, so we need a special save function
            save_zero_three_model(model, tokenizer, args)
        print_rank_0('save successfully!', args.global_rank)


if __name__ == "__main__":
    main()
