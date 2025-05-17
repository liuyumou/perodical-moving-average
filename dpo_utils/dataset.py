from torch.utils.data import Dataset, IterableDataset
from datasets import load_dataset, Features
from transformers import GPT2Tokenizer, GPT2TokenizerFast
import datasets
import torch
from tqdm import tqdm
import random
import json
import os
from copy import deepcopy
# from tokenizer import TiktokenTokenizer
import numpy as np


class DahoasSFTStaticPromptsDataset(Dataset):

    def __init__(self,
                 block_size,
                 max_examples=None,
                 tokenizer=None) -> None:
        super().__init__()
        dataset = load_dataset("Dahoas/rm-static", split="train")
        self.prompts = []

        # if tokenizer_name == "huggingface/gpt2":
        #     tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        #     tokenizer.pad_token = tokenizer.eos_token
        # elif tokenizer_name == "huggingface/gpt2fast":
        #     tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        #     tokenizer.pad_token = tokenizer.eos_token
        # elif tokenizer_name == "tiktoken/gpt2":
        #     tokenizer = TiktokenTokenizer('gpt2')
        assert tokenizer is not None

        cnt = 0
        print(f"Loading DahoasSFTStaticPromptsDataset")
        for data in dataset:
            cnt += 1
            prompt = data['prompt']
            tokens = tokenizer(prompt,
                               max_length=block_size,
                               padding="max_length",
                               truncation=True,
                               return_tensors="pt")

            self.prompts.append(
                [tokens['input_ids'], tokens['attention_mask'], torch.sum(tokens['attention_mask'])])

            if max_examples and cnt >= max_examples:
                break

    @classmethod
    def save(cls, split, fp):
        dataset = load_dataset("fka/awesome-chatgpt-prompts", split=split)
        examples = []
        for data in tqdm(dataset):
            examples.append(data["prompt"])
        import json
        json.dump(examples, fp)

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx][0], self.prompts[idx][1], self.prompts[idx][2]  # (1, T), (1, T)

class RLHFDataset(Dataset):
    """
    https://huggingface.co/datasets/Anthropic/hh-rlhf#dataset-summary
    """

    def __init__(self,
                 data_path,
                 block_size,
                 split='train',
                 max_examples=None,
                 tokenizer=None) -> None:
        super().__init__()
        dataset = datasets.load_from_disk('./dataset/'+data_path)
        dataset = dataset[split]
        self.pairs = []
        self.masks = []

        assert tokenizer is not None

        # torch.manual_seed(123) # ensure consistent dataset split
        num_data = len(dataset) // 2
        sub_dset = dataset

        print(f"Loading RLHF Dataset...")
        for i in range(num_data):
            indices = []
            masks = []
            for split in ["chosen", "rejected"]:
                out = tokenizer(sub_dset[int(i)]['prompt']+sub_dset[int(i)][split],
                                 max_length=block_size,
                                 padding="max_length",
                                 truncation=True,
                                 return_tensors="pt")
                indices.append(out["input_ids"].squeeze())
                masks.append(out["attention_mask"].squeeze())
            self.pairs.append(torch.stack(indices, dim=0))
            self.masks.append(torch.stack(masks, dim=0))
            
            if max_examples and i >= max_examples:
                break

    @classmethod
    def save(cls, split, fp):
        dataset = load_dataset("Anthropic/hh-rlhf", split=split)
        examples = []
        for data in tqdm(dataset):
            examples.append(data["chosen"])
        import json
        json.dump(examples, fp)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx], self.masks[idx]  # (2, T), (2, T)

class SFTDataset(Dataset):
    def __init__(self,
                 data_path,
                 block_size,
                 split='train',
                 max_examples=None,
                 tokenizer=None) -> None:
        super().__init__()
        save = False
        
        dataset = datasets.load_from_disk('./dataset/'+data_path)
        dataset = dataset[split]
        sft_idx = np.random.choice(len(dataset), len(dataset)//2, replace=False)

        self.prompt_list = []
        self.attn_mask_list = []
        self.labels = []
        self.block_size = block_size
        assert tokenizer is not None
        IGNORE_INDEX = -100

        cnt = 0
        print(f"Loading SFT {split} split")
        for idx in sft_idx:
            cnt += 1
            prompt = dataset[int(idx)]['prompt']
            prompt_token = tokenizer(
                    prompt,
                    max_length=block_size,
                    padding="do_not_pad",
                    truncation=True,
                    return_tensors="pt",
                )
            prompt_length = len(prompt_token["input_ids"].flatten())
            chosen = prompt+dataset[int(idx)]['chosen']
            # chosen = dataset[int(idx)]['chosen']
            # response_text = chosen + "<|endoftext|>"
            response = tokenizer(chosen, 
                                 max_length=block_size,
                                 padding="max_length",
                                 truncation=True,
                                 return_tensors="pt")

            self.prompt_list.append(response["input_ids"].squeeze())
            self.attn_mask_list.append(response["attention_mask"].squeeze())
            
            labels = deepcopy(response["input_ids"])
            if prompt_length == block_size:
                # drop samples that are all of prompts
                # filtered += 1
                continue
            else:
                labels[:prompt_length] = IGNORE_INDEX
            labels[(1 - response["attention_mask"]).to(torch.bool)] = (
                IGNORE_INDEX
            )

            response["labels"] = labels
            self.labels.append(response["labels"].squeeze(0))
            
            
            if max_examples and cnt >= max_examples:
                break

        print(f"Loaded {len(self.prompt_list)} pieces from {cnt} examples.")
        
    def __len__(self):
        return len(self.prompt_list)

    def __getitem__(self, idx):
        return self.prompt_list[idx], self.attn_mask_list[idx], self.labels[idx]  # (T), (T), (T)

class EYLSFTStaticDataset(Dataset):

    def __init__(self,
                 block_size,
                 split='train',
                 max_examples=None,
                 tokenizer=None) -> None:
        super().__init__()
        if split == "train":
            with open("./sft_train.json") as fp:
                dataset = json.load(fp)
        else:
            with open("./sft_test.json") as fp:
                dataset = json.load(fp)
        self.tokens = []
        self.block_size = block_size
        # if tokenizer_name == "huggingface/gpt2":
        #     tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        #     tokenizer.pad_token = tokenizer.eos_token
        # elif tokenizer_name == "huggingface/gpt2fast":
        #     tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        # elif tokenizer_name == "tiktoken/gpt2":
        #     tokenizer = TiktokenTokenizer('gpt2')
        assert tokenizer is not None

        cnt = 0
        print(f"Loading EYLSFTStaticDataset {split} split")
        for chosen in dataset:
            cnt += 1
            response_text = chosen + "<|endoftext|>"
            response = tokenizer(response_text)

            self.tokens += response['input_ids']
            if max_examples and cnt >= max_examples:
                break

        self.tokens = torch.tensor(self.tokens, dtype=torch.long)
        print(f"Loaded {len(self.tokens)} tokens from {cnt} examples.")

    def __len__(self):
        import sys
        return sys.maxsize

    def __getitem__(self, idx):
        start = random.randint(0, len(self.tokens) - self.block_size - 2)
        x = self.tokens[start:start + self.block_size]
        y = self.tokens[start + 1:start + self.block_size + 1]
        return x, y


class DahoasSFTStaticDataset(IterableDataset):
    """
    https://huggingface.co/datasets/Dahoas/sft-static
    """

    def __init__(self,
                 block_size,
                 split='train',
                 max_examples=None,
                 tokenizer=None) -> None:
        super().__init__()
        dataset = load_dataset(
            "Dahoas/sft-static",
            revision="90e35d9cd625075f1224c4241734716ec9f0db78",
            split=split)
        self.tokens = []
        self.block_size = block_size

        # if tokenizer_name == "huggingface/gpt2":
        #     tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        #     tokenizer.pad_token = tokenizer.eos_token
        # elif tokenizer_name == "huggingface/gpt2fast":
        #     tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        # elif tokenizer_name == "tiktoken/gpt2":
        #     tokenizer = TiktokenTokenizer('gpt2')
        assert tokenizer is not None

        cnt = 0
        print(f"Loading DahoasSFTStaticDataset {split} split")
        for data in dataset:
            cnt += 1
            prompt = data['prompt']

            response_text += prompt + data['response'] + "<|endoftext|>"
            response = tokenizer(response_text)

            self.tokens += response['input_ids']
            if max_examples and cnt >= max_examples:
                break

        self.tokens = torch.tensor(self.tokens, dtype=torch.long)

    def __iter__(self):
        start = random.randint(0, len(self.tokens) - self.block_size - 2)
        x = self.tokens[start:start + self.block_size]
        y = self.tokens[start + 1:start + self.block_size + 1]
        yield x, y


class DahoasRMStaticDataset(Dataset):
    """
    https://huggingface.co/datasets/Dahoas/rm-static
    """

    def __init__(self,
                 block_size,
                 split='train',
                 max_examples=None,
                 tokenizer=None) -> None:
        super().__init__()
        dataset = load_dataset("Dahoas/rm-static", split=split)
        self.pairs = []
        self.masks = []

        # if tokenizer_name == "huggingface/gpt2":
        #     tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        #     tokenizer.pad_token = tokenizer.eos_token
        # elif tokenizer_name == "huggingface/gpt2fast":
        #     tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        # elif tokenizer_name == "tiktoken/gpt2":
        #     tokenizer = TiktokenTokenizer('gpt2')
        assert tokenizer is not None

        cnt = 0
        print(f"Loading DahoasRMStaticDataset {split} split")
        for data in dataset:
            cnt += 1
            prompt = data['prompt']

            positive_text = prompt + data['chosen'] + "<|endoftext|>"
            positive = tokenizer(positive_text,
                                 max_length=block_size,
                                 padding="max_length",
                                 truncation=True,
                                 return_tensors="pt")

            negative_text = prompt + data['rejected'] + "<|endoftext|>"
            negative = tokenizer(negative_text,
                                 max_length=block_size,
                                 padding="max_length",
                                 truncation=True,
                                 return_tensors="pt")

            self.pairs.append(
                torch.stack((positive['input_ids'], negative['input_ids']),
                            dim=0))

            self.masks.append(
                torch.stack(
                    (positive['attention_mask'], negative['attention_mask']),
                    dim=0))
            if max_examples and cnt >= max_examples:
                break

    @classmethod
    def save(cls, split, fp):
        dataset = load_dataset("Dahoas/rm-static", split=split)
        examples = []
        for data in tqdm(dataset):
            examples.append(data["prompt"] + data["chosen"])
        import json
        json.dump(examples, fp)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx], self.masks[idx]  # (2, T), (2, T)


class AnthropicHHRLHFDataset(Dataset):
    """
    https://huggingface.co/datasets/Anthropic/hh-rlhf#dataset-summary
    """

    def __init__(self,
                 block_size,
                 split='train',
                 max_examples=None,
                 tokenizer=None) -> None:
        super().__init__()
        dataset = load_dataset("Anthropic/hh-rlhf", split=split)
        self.pairs = []
        self.masks = []

        # if tokenizer_name == "huggingface/gpt2":
        #     tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        #     tokenizer.pad_token = tokenizer.eos_token
        # elif tokenizer_name == "huggingface/gpt2fast":
        #     tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        # elif tokenizer_name == "tiktoken/gpt2":
        #     tokenizer = TiktokenTokenizer('gpt2')
        assert tokenizer is not None
        
        cnt = 0
        for data in dataset:
            positive = tokenizer(data["chosen"],
                                 max_length=block_size,
                                 padding="max_length",
                                 truncation=True,
                                 return_tensors="pt")
            positive_indices = positive["input_ids"]
            positive_mask = positive["attention_mask"]

            negative = tokenizer(data["rejected"],
                                 max_length=block_size,
                                 padding="max_length",
                                 truncation=True,
                                 return_tensors="pt")
            negative_indices = negative["input_ids"]
            negative_mask = negative["attention_mask"]

            self.pairs.append(
                torch.stack((positive_indices, negative_indices), dim=0))

            self.masks.append(
                torch.stack((positive_mask, negative_mask), dim=0))
            cnt += 1
            if max_examples and cnt >= max_examples:
                break

    @classmethod
    def save(cls, split, fp):
        dataset = load_dataset("Anthropic/hh-rlhf", split=split)
        examples = []
        for data in tqdm(dataset):
            examples.append(data["chosen"])
        import json
        json.dump(examples, fp)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx], self.masks[idx]  # (2, T), (2, T)
