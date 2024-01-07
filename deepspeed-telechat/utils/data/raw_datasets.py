# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0


from datasets import load_dataset

class PromptRawDataset(object):

    def __init__(self, output_path, seed, local_rank, dataset_name):
        self.output_path = output_path
        self.seed = seed
        self.local_rank = local_rank
        self.raw_datasets = load_dataset(path="json",data_files=dataset_name)


    def get_train_data(self):
        return

    def get_eval_data(self):
        return

    def get_prompt(self, sample):
        return

    def get_prompt_and_answer(self, sample):
        return



class TelechatDataset(PromptRawDataset):

    def __init__(self, output_path, seed, local_rank, dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name)
        self.dataset_name = dataset_name


    def get_train_data(self):
        dataset = self.raw_datasets["train"]
        return dataset

    def get_eval_data(self):
        dataset = self.raw_datasets["train"]
        return dataset

    def get_prompt(self, sample):
        return "<_user>" + sample['input'] + "<_bot>"

    def get_prompt_and_answer(self, sample):
        return "<_user>" + sample['input'] + "<_bot>" + sample['output'] + "<_end>"

