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
import json
import re
import math
import random
from multiprocessing import Pool
from functools import partial
from itertools import chain

def get_raw_dataset(dataset_name, output_path, seed):
    return raw_datasets.TelechatDataset(output_path, seed, dataset_name)


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

def get_weight_data(current_dataset, dataset_weight):
    dataset = []
    all_lines = []
    for i, tmp_data in enumerate(current_dataset):
        if dataset_weight < 1.0 and random.random() > dataset_weight: continue
        input = tmp_data['input']
        input = re.sub(r"^<_user>", "", input, flags=re.S)
        input = "<_user>" + input
        output = tmp_data['output']
        output = re.sub(r"^<_bot>", "", output, flags=re.S)
        if "<_bot>" in input:  ### multiturn
            concat_line = ""
            input_turns = input.split("<_user>")[1:]
            for item in input_turns:
                if "<_bot>" in item:
                    concat_line += "<_user>" + item + "<_end>"
                else:
                    concat_line += "<_user>" + item + "<_bot>"
            concat_line += output + "<_end>"
        else:  ####single turn
            concat_line = str(input) + "<_bot>" + str(output) + "<_end>"
        assert concat_line.count("<_user>") == concat_line.count("<_bot>") == concat_line.count("<_end>")
        if dataset_weight < 1.0:
            all_lines.append(concat_line)
        else:
            weight_integer = math.floor(dataset_weight)
            weight_decimal = dataset_weight - weight_integer
            for i in range(math.floor(dataset_weight)):
                all_lines.append(concat_line)
            if random.random() < weight_decimal:
                all_lines.append(concat_line)
    return all_lines

def create_dataset( dataset_name, dataset_weight, output_path, seed):
    raw_dataset = get_raw_dataset(dataset_name, output_path, seed)
    train_dataset = raw_dataset.get_train_data()
    train_dataset = get_weight_data(train_dataset, dataset_weight)
    return train_dataset

def process_concat_data(text, tokenizer, max_seq_len, args):
    texts = text.split("<_end>")
    sentence_ids = []
    for text in texts:
        if text != '':
            input, output = text.split("<_bot>")
            input = re.sub(r"^<_user>", "", input, flags=re.S)
            input_ids = [args.user_token_id] + tokenizer(input)["input_ids"]
            output_ids = [args.bot_token_id] + tokenizer(output)["input_ids"] + [args.end_token_id]
            sentence_ids += (input_ids + output_ids)
    sentence_ids = [3] * (max_seq_len - len(sentence_ids)) + sentence_ids
    return {"input_ids": torch.tensor(sentence_ids), "attention_mask": torch.ones(len(sentence_ids))}


def process(id, samples, tokenizer, max_seq_len, num_workers, num_samples, args):
    cnt = 0
    sample_nums = num_samples
    all_lines = []
    dataset = []
    while cnt < sample_nums // num_workers:
        index = id
        single_process_length = len(samples) // num_workers
        #### 统计所有句子的长度
        lengths = []
        chunk_size = 1
        all_lines_shard = samples[index * single_process_length:(index + 1) * single_process_length] if index < num_workers - 1 \
            else samples[index * single_process_length:]
        all_lines_chunk_list = [all_lines_shard[i:i + chunk_size] for i in range(0, len(all_lines_shard), chunk_size)]
        for i in tqdm(range(len(all_lines_chunk_list))):
            encoded_batch = tokenizer.batch_encode_plus(all_lines_chunk_list[i], padding=False)
            for j in range(len(encoded_batch["input_ids"])):
                lengths.append(len(encoded_batch["input_ids"][j]))
        all_lines_and_length = []
        for i, item in tqdm(enumerate(all_lines_shard)):
            if lengths[i] < max_seq_len - 10:  ###只有小于maxlen的才可以被处理
                all_lines_and_length.append((item, lengths[i]))

        pool = all_lines_and_length
        min_threshold = min(lengths)
        pad_count = 0
        tot = 0
        pbar = tqdm(total=len(pool), desc=f"Processing {id}, Concating dataset", disable=(id != 0))
        while pool:
            ptr = 0
            buffer_len = 0
            buffer = []
            while ptr < len(pool) and (max_seq_len - buffer_len) > min_threshold:
                if pool[ptr][1] + buffer_len < max_seq_len - 10:  ####至少留10个padding
                    buffer_len += pool[ptr][1]
                    buffer.append(pool[ptr][0])
                    pool.pop(ptr)
                    pbar.update(1)
                else:
                    ptr += 1
            buffer_text = "".join(buffer)
            output = buffer_text
            pad_count += (max_seq_len - buffer_len)
            tot += 1
            assert output.count("<_user>") == output.count("<_bot>") == output.count("<_end>")
            if output.count("<_user>") == output.count("<_bot>") == output.count("<_end>") and output.count(
                    "<_user>") >= 1:
                all_lines.append(output)
                cnt += 1
                if cnt >= sample_nums // num_workers: break
        pbar.close()
    for line in tqdm(all_lines, desc="Convert token ids", disable=(id != 0)):
        tokens = process_concat_data(line, tokenizer, max_seq_len, args)
        dataset.append(tokens)
    return dataset

def create_prompt_dataset(data_path,
                          output_path,
                          seed,
                          tokenizer,
                          max_seq_len,
                          num_workers,
                          num_samples,
                          process_method,
                          args):
    """
    Creates the dataset
    """
    os.makedirs(output_path, exist_ok=True)
    train_fname = f"{output_path}/train_data.pt"
    print(f"train_fname:{train_fname}")

    with open(data_path, "r", encoding="utf-8") as f: data_dic = json.load(f)
    train_datasets = []
    train_size = 0
    for dataset_name, dataset_weight in data_dic.items():
        train_dataset = create_dataset(
            dataset_name, dataset_weight,
            output_path, seed)
        train_datasets.extend(train_dataset)
        train_size += len(train_dataset)
    shuffle(train_datasets)
    if process_method == "multiple":
        with Pool(processes=num_workers) as pool:
            partial_process = partial(process, samples=train_datasets,
                                      tokenizer=tokenizer, max_seq_len=max_seq_len,
                                      num_workers=num_workers, num_samples=num_samples, args=args)
            results = pool.map(partial_process, [i for i in range(num_workers)])
        combined_results = list(chain.from_iterable(results))
    else:
        combined_results = process(0, train_datasets, tokenizer, max_seq_len, num_workers, 0, args)
    train_dataset = PromptDataset(combined_results)
    torch.save(train_dataset, train_fname)



