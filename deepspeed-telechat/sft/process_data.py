#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import sys

from transformers import AutoTokenizer


sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils.data.data_utils import create_prompt_dataset
from utils.utils import set_random_seed

def parse_args():
    parser = argparse.ArgumentParser(
        description=
        "Finetune a transformers model on a causal language modeling task")
    parser.add_argument('--data_path',
                        type=str,
                        required=True,
                        help='A json file store dataset path and weight')
    parser.add_argument(
        '--data_output_path',
        type=str,
        default='/tmp/data_files/',
        help='Where to save the processed data.'
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        help=
        "Path to the tokenizer",
        required=True,
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=512,
        help="The maximum sequence length.",
    )
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="A seed for reproducible training.")
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
    parser.add_argument("--num_workers",
                        type=int,
                        default=5,
                        help="Number of workers when tokenizing dataset")
    parser.add_argument("--num_samples",
                        type=int,
                        required=True,
                        help="Number of samples while training")
    parser.add_argument('--process_method',
                        choices=['single', 'multiple'],
                        required=True,
                        help='Choose the method (multiple process or single process) while processing dataset, note that'
                             'when using both multi-process and multi-nodes, you should have a shared system.')
    parser.add_argument("--gradient_checkpointing_use_reentrant", action="store_true")
    args = parser.parse_args()

    return args

def load_telechat_tokenizer(tokenizer_path, fast_tokenizer=True):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path,
                                              fast_tokenizer=fast_tokenizer,
                                              padding_side="left",
                                              trust_remote_code=True)
    return tokenizer

def main():
    args = parse_args()
    set_random_seed(args.seed)
    tokenizer = load_telechat_tokenizer(args.tokenizer_path, fast_tokenizer=True)
    args.user_token_id = tokenizer.convert_tokens_to_ids(args.user_token)
    args.bot_token_id = tokenizer.convert_tokens_to_ids(args.bot_token)
    args.end_token_id = tokenizer.convert_tokens_to_ids(args.end_token)

    create_prompt_dataset(
        args.data_path,
        args.data_output_path,
        args.seed,
        tokenizer,
        args.max_seq_len,
        args.num_workers,
        args.num_samples,
        args.process_method,
        args)

    print("Finish processing data!")


if __name__ == "__main__":
    main()
