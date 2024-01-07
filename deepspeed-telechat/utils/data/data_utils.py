# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

"""
Part of the code was adopted from https://github.com/microsoft/Megatron-DeepSpeed/blob/main/megatron/data/dataset_utils.py
"""
import torch
from torch.utils.data import Dataset, Subset, ConcatDataset
import numpy as np
import os
import hashlib
from . import raw_datasets
from tqdm import tqdm
from random import shuffle


def get_raw_dataset(dataset_name, output_path, seed, local_rank):
    return raw_datasets.TelechatDataset(output_path, seed, local_rank, dataset_name)


def get_shuffle_idx(seed, size):
    np_rng = np.random.RandomState(seed=seed)
    dtype_ = np.uint32
    if size >= (np.iinfo(np.uint32).max - 1):
        dtype_ = np.int64
    shuffle_idx = np.arange(start=0, stop=size, step=1, dtype=dtype_)
    np_rng.shuffle(shuffle_idx)
    return shuffle_idx


class PromptDataset(Dataset):

    def __init__(self, chosen_dataset) -> None:
        super().__init__()
        self.dataset = chosen_dataset

    def __len__(self):
        length = len(self.dataset)
        return length

    def __getitem__(self, idx):
        return {
            "input_ids": self.dataset[idx]["input_ids"],
            "attention_mask": self.dataset[idx]["attention_mask"],
            "labels": self.dataset[idx]["input_ids"]
        }


def process_dataset(current_dataset, tokenizer, max_seq_len):
    dataset = []
    all_lines = []
    for i, tmp_data in enumerate(current_dataset):
        input = tmp_data['input']
        if not input.startswith("<_user>"):
            input = "<_user>" + input
        output = tmp_data['output']
        if "<_bot>" in input: ### multiturn
            concat_line = ""
            input_turns = input.split("<_user>")[1:]
            for item in input_turns:
                if "<_bot>" in item:
                    concat_line += "<_user>" + item + "<_end>"
                else:
                    concat_line += "<_user>" + item + "<_bot>"
            concat_line += output + "<_end>"
        else: ####single turn
            concat_line = str(input) + "<_bot>" + str(output) + "<_end>"
        assert concat_line.count("<_user>") == concat_line.count("<_bot>") == concat_line.count("<_end>")
        all_lines.append(concat_line)
    shuffle(all_lines)
    previous_corpus_token_cnt = 0
    shard = []
    padding_out = []
    for corpus in tqdm(all_lines):
        corpus_ids = tokenizer(corpus, return_tensors="pt")
        if previous_corpus_token_cnt + len(corpus_ids["input_ids"][0]) < max_seq_len:
            shard.append(corpus)
            previous_corpus_token_cnt += len(corpus_ids["input_ids"][0])
        else:
            shard_output = "".join(shard)
            shard_output = (max_seq_len - previous_corpus_token_cnt) * "<pad>" + shard_output
            assert len(tokenizer(shard_output, return_tensors="pt")["input_ids"][0]) == max_seq_len
            if shard_output.count("<_user>") >= 1:
                padding_out.append(shard_output)
            if len(corpus_ids["input_ids"][0]) < max_seq_len:
                shard = [corpus]
                previous_corpus_token_cnt = len(corpus_ids["input_ids"][0])
            else:
                shard = []
                previous_corpus_token_cnt = 0
    print("prompt length: ",len(padding_out))
    for dt in padding_out:
        tokens = tokenizer(dt,return_tensors="pt")
        tokens["input_ids"] = tokens["input_ids"].squeeze(0)
        tokens["attention_mask"] = tokens["attention_mask"].squeeze(0)
        dataset.append(tokens)
    return PromptDataset(dataset)

def create_dataset(local_rank, dataset_name, output_path, seed, tokenizer, max_seq_len):
    raw_dataset = get_raw_dataset(dataset_name, output_path, seed, local_rank)
    train_dataset = raw_dataset.get_train_data()
    train_dataset = process_dataset(train_dataset, tokenizer, max_seq_len)

    return train_dataset


def create_prompt_dataset(local_rank,
                          data_path,
                          output_path,
                          seed,
                          tokenizer,
                          max_seq_len,
                          use_cache_dataset):
    """
    Creates the prompt dataset
    """
    os.makedirs(output_path, exist_ok=True)
    fname = "_".join(data_path)
    tokenizer_name = tokenizer.init_kwargs["name_or_path"].replace("/", "_")
    fname = f"{fname}_seed{seed}_tokenizer{tokenizer_name}_seqlen{max_seq_len}"
    fname = "_".join(fname.split("/"))
    fname = hashlib.sha256(fname.encode()).hexdigest(
    )  # hash the file name to avoid too long file name
    train_fname = f"{output_path}/traindata_{fname}.pt"
    print(f"train_fname:{train_fname}")

    if use_cache_dataset:
        cache_found = os.path.isfile(train_fname)
        buf_create_cache = torch.ByteTensor([not cache_found]).cuda()
        torch.distributed.all_reduce(buf_create_cache)
    else:
        buf_create_cache = torch.ByteTensor([True]).cuda()
        torch.distributed.all_reduce(buf_create_cache)

    if local_rank <= 0 and buf_create_cache.item() != 0:
        if len(data_path) == 1:  # Single dataset.
            train_dataset = create_dataset(
                local_rank, data_path[0], output_path,
                seed, tokenizer, max_seq_len)
        else:  # Blending datasets.
            train_datasets = []
            train_size = 0
            for d_path in data_path:
                train_dataset = create_dataset(
                    local_rank, d_path, output_path,
                    seed, tokenizer, max_seq_len)
                train_datasets.append(train_dataset)
                train_size += len(train_dataset)
            train_dataset = ConcatDataset(train_datasets)
            shuffle_idx = get_shuffle_idx(seed, train_size)
            train_dataset = Subset(train_dataset, shuffle_idx.tolist())
        torch.save(train_dataset, train_fname)
    torch.distributed.barrier()
    return torch.load(train_fname)



