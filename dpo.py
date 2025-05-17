#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import argparse
import os
import sys
import random
from tqdm import tqdm
import gc
import time
import math
import json
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F
import numpy as np

from torch.utils.tensorboard import SummaryWriter

from transformers import (
    SchedulerType,
    default_data_collator,
    AutoModelForCausalLM,
    get_scheduler,
)

import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from optimizer.AdamW_PMA import AdamW_PMA
from optimizer.Lion_PMA import Lion_PMA

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)

# from dschat.remax.remax_trainer import (
#     DeepSpeedReMaxTrainer,
#     DeepSpeedReMaxTrainerUnsupervised,
# )
# from dschat.remax.rlhf_engine import DeepSpeedRLHFEngine
# from dschat.remax.perf import print_throughput_step3
# from dschat.remax.data_utils import create_rl_prompt_dataset

from utils.data.data_utils import (
    create_prompt_dataset,
    MiniDataset,
    DataCollatorRLHF,
    get_unsupervised_data,
    DataCollatorReward,
)
from utils.utils import (
    print_rank_0,
    to_device,
    save_hf_format,
    set_random_seed,
    get_all_reduce_mean,
    moving_average,
    save_zero_three_model,
    load_hf_tokenizer,
    ExponentialMovingAverage,
    get_optimizer_grouped_parameters,
)
from utils.model.model_utils import create_hf_model
from utils.module.lora import convert_lora_to_linear_layer
from utils.io_utils import save_code, print_machine_info
from utils.ds_utils import get_train_ds_config
from utils.perf import print_throughput, calculate_flops, get_hf_configs
from deepspeed.accelerator import get_accelerator

def parse_args():
    global writer
    parser = argparse.ArgumentParser(description="(Step 3) RLHF training arguments")

    parser.add_argument("--algo", type=str, default="remax", help="Algorithm name")
    parser.add_argument(
        "--data_path",
        nargs="*",
        default=["Dahoas/rm-static"],
        help="Path to the training dataset. Accepted format: 1) a single data path, 2) multiple datasets in the form: dataset1-path dataset2-path ...",
    )
    parser.add_argument(
        "--prompt_data_source",
        type=str,
        default=None,
        choices=["reward", None],
        help="Prompt data source.",
    )
    parser.add_argument(
        "--data_split",
        type=str,
        default="2,4,4",
        help="Comma-separated list of proportions for training phase 1, 2, and 3 data. For example the split `2,4,4` "
        "will use 60%% of data for phase 1, 20%% for phase 2 and 20%% for phase 3.",
    )
    parser.add_argument(
        "--max_size", type=int, default=int(1e6), help="Max size for training prompts."
    )
    parser.add_argument(
        "--data_output_path",
        type=str,
        default="/tmp/data_files",
        help="Where to store the data-related files such as shuffle index. This needs to be on a local storage of a node (not on a shared storage)",
    )
    parser.add_argument(
        "--unsupervised_dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--unsupervised_dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--unsup_coef",
        type=float,
        default=27.8,
        help="""gamma in Equation 2 from InstructGPT paper""",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["adamw", "adam_mini", "lion", "adamw_pma", "lion_pma"],
        help="Optimizer.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--num_padding_at_beginning",
        type=int,
        default=0,
        help="OPT model has a fixed number (1) of padding tokens at the beginning of the input. We did not see this in other models but keep it as an option for now.",
    )
    parser.add_argument(
        "--per_device_generation_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader and generation purpose.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Mini Batch size (per device) for the training dataloader and training purpose.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=16,
        help="Mini Batch size (per device) for the evaluation dataloader and evaluation purpose.",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=512,
        help="The maximum sequence length.",
    )
    parser.add_argument(
        "--generation_batches",
        type=int,
        default=1,
        help="Generate x batches to go to training mode.",
    )
    parser.add_argument(
        "--max_prompt_seq_len",
        type=int,
        default=256,
        help="The maximum sequence length.",
    )
    parser.add_argument(
        "--max_answer_seq_len",
        type=int,
        default=256,
        help="The maximum sequence length.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=9.65e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight decay to use."
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",
        help="The scheduler type to use.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=100,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Where to store the model."
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--data_seed",
        type=int,
        default=1234,
        help="A seed for reproducible data processing.",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local_rank for distributed training on gpus",
    )

    # DeepSpeed
    parser.add_argument(
        "--enable_hybrid_engine",
        action="store_true",
        help="Enable hybrid engine for actor model to optimize both inference and training through DeepSpeed.",
    )
    parser.add_argument(
        "--unpin_actor_parameters",
        action="store_true",
        help="Unpin actor's parameters during generation. This makes generation slower but requires less memory.",
    )
    parser.add_argument(
        "--release_inference_cache",
        action="store_true",
        help="Release the memory cache used for inference. This makes generation preparation slower but might increase e2e throughput by using larger batch size.",
    )
    parser.add_argument(
        "--inference_tp_size",
        type=int,
        default=1,
        help="Tensor-parallelism degree used for the inference-optimization. Please note hybrid-engine need to be enabled when using this feature.",
    )
    parser.add_argument(
        "--tp_gather_partition_size",
        type=int,
        default=8,
        help="Granularity to bring in layers for TP sharding inside the hybrid engine. Please note hybrid-engine and tp_inference_size > 1 need to be true when using this feature.",
    )
    parser.add_argument(
        "--flash_attn", type=bool, default=True, help="Enable flash attention."
    )
    parser.add_argument(
        "--offload", action="store_true", help="Enable ZeRO Offload techniques."
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp16",
        choices=["fp16", "bf16", "fp32"],
        help="Training data type",
    )
    parser.add_argument(
        "--offload_reference_model",
        action="store_true",
        help="Enable ZeRO Offload techniques for reference model",
    )
    parser.add_argument(
        "--offload_reward_model",
        action="store_true",
        help="Enable ZeRO Offload techniques for reward model",
    )
    parser.add_argument(
        "--actor_zero_stage",
        type=int,
        default=0,
        help="ZeRO optimization stage for Actor model.",
    )
    parser.add_argument(
        "--reference_zero_stage",
        type=int,
        default=0,
        help="ZeRO optimization stage for Reference model.",
    )
    parser.add_argument(
        "--reward_zero_stage",
        type=int,
        default=0,
        help="ZeRO optimization stage for reward model.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable HF gradient checkpointing for Actor model.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="If actor dropout configured, use it. "
        "Otherwise, keep the default dropout configuration of the actor model.",
    )
    parser.add_argument(
        "--reward_dropout",
        type=float,
        default=None,
        help="If actor dropout configured, use it. "
        "Otherwise, keep the default dropout configuration of the reward model.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temeperature for sampling.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top p for sampling.",
    )
    parser.add_argument(
        "--penalty",
        type=str,
        default="kl_onestep",
        choices=["kl", "kl_onestep", "entropy", "entropy_onestep"],
        help="Penalty type.",
    )
    parser.add_argument(
        "--kl_ctl", type=float, default=0.1, help="KL penalty coefficient."
    )
    parser.add_argument(
        "--kl_with_baseline", action="store_true", help="KL with baseline value"
    )
    parser.add_argument(
        "--clip_reward_value", type=float, default=5.0, help="Reward clip coefficient."
    )
    parser.add_argument(
        "--clip_kl_value", type=float, default=0.0, help="KL clip coefficient."
    )
    parser.add_argument(
        "--dpo_beta", type=float, default=1.0, help="DPO beta coefficient."
    )
    ## LoRA for efficient training setting
    parser.add_argument(
        "--lora_dim",
        type=int,
        default=0,
        help="If > 0, use LoRA for efficient training.",
    )
    parser.add_argument(
        "--actor_lora_module_name",
        type=str,
        default="decoder.layers.",
        help="The scope of LoRA.",
    )
    parser.add_argument(
        "--only_optimize_lora",
        action="store_true",
        help="Only optimize the LoRA parameters.",
    )
    parser.add_argument(
        "--actor_lora_learning_rate",
        type=float,
        default=5e-4,
        help="Initial actor LoRA learning rate (after the potential warmup period) to use.",
    )
    ## Make EMA as an optional feature
    parser.add_argument(
        "--enable_ema", action="store_true", help="Enable EMA checkpoint for the model."
    )
    ## Mixed Precision ZeRO++
    parser.add_argument(
        "--enable_mixed_precision_lora",
        action="store_true",
        help="Enable Mixed Precision ZeRO++ for training and generation.",
    )
    ## low precision
    parser.add_argument(
        "--compute_fp32_loss",
        action="store_true",
        help="Relevant for low precision dtypes (fp16, bf16, etc.). "
        "If specified, loss is calculated in fp32."
        "This applies for actor model.",
    )
    ## Tensorboard logging
    parser.add_argument(
        "--enable_tensorboard", action="store_true", help="Enable tensorboard logging"
    )
    parser.add_argument("--tensorboard_path", type=str, default="step3_tensorboard")
    ## Print actor model answers during training
    parser.add_argument(
        "--print_answers",
        action="store_true",
        help="Print prompt and answers during training",
    )
    parser.add_argument(
        "--print_answers_interval",
        type=int,
        default=20,
        help="If --print_answers enabled, controls the printing interval.",
    )
    parser.add_argument(
        "--save_answers",
        action="store_true",
        help="Save prompt and answers during training",
    )
    parser.add_argument(
        "--save_model",
        action="store_true",
        help="Wether to save model checkpoint.",
    )
    parser.add_argument(
        "--eval_samples", type=int, default=1000, help="Maximum evaluation samples"
    )
    ## template
    parser.add_argument(
        "--actor_template",
        type=str,
        default="none",
        help="Prompt template for Actor model.",
    )
    parser.add_argument(
        "--reward_template",
        type=str,
        default="none",
        help="Prompt template for reward model.",
    )
    parser.add_argument("--eval_interval", type=int, default=10, help="Eval interval")
    ## Testing
    parser.add_argument(
        "--enable_test_mode",
        action="store_true",
        help="Enable a testing mode that terminates training based on args.test_stop_step",
    )
    parser.add_argument(
        "--test_stop_step",
        type=int,
        default=0,
        help="Training non-overflow step at which to terminate training during testing.",
    )
    parser.add_argument(
        "--print_loss", action="store_true", help="Prints loss at each step."
    )

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    # Validate settings
    if args.inference_tp_size > 1:
        assert (
            args.actor_zero_stage == 3
        ), "Zero stage 3 must be used to do Tensor sharding in the hybrid engine"

    if (
        args.actor_zero_stage == 2
        and args.enable_hybrid_engine
        and args.offload
        and args.lora_dim == 0
    ):
        pass
        # raise ValueError(
        #     "The combination of [actor_zero_stage==2, enable_hybrid_engine=True, offload=True, lora=False] is currently unsupported due to training instability!"
        # )

    return args


def create_datasets(args, tokenizer, train_phase=2):
    unsupervised_training_enabled = (
        args.unsupervised_dataset_name and args.unsupervised_dataset_config_name
    )
    if args.prompt_data_source == "reward":
        from dpo_utils.data_utils import create_rl_prompt_dataset
        prompt_train_dataset, prompt_eval_dataset = create_rl_prompt_dataset(
            args.local_rank,
            args.data_path,
            args.data_split,
            args.data_output_path,
            train_phase,
            args.data_seed,
            tokenizer,
            args.max_prompt_seq_len,
            end_of_conversation_token=tokenizer.eos_token,
            reload=True,
            template=args.actor_template,
            max_size=args.max_size,
        )
        # raise NotImplementedError("Reward model is not supported yet.")
    else:
        prompt_train_dataset, prompt_eval_dataset = create_prompt_dataset(
            args.local_rank,
            args.data_path,
            args.data_split,
            args.data_output_path,
            train_phase,
            args.data_seed,
            tokenizer,
            args.max_prompt_seq_len,
            end_of_conversation_token=tokenizer.eos_token,
            reload=True,
            template=args.actor_template,
            max_size=args.max_size,
        )
    # print('train_dataset',len(prompt_train_dataset))
    # print('eval_dataset',len(prompt_eval_dataset))
    _, ppl_eval_dataset = create_prompt_dataset(
        args.local_rank,
        args.data_path,
        args.data_split,
        args.data_output_path,
        1,
        args.data_seed,
        tokenizer,
        args.max_prompt_seq_len + args.max_answer_seq_len,
        end_of_conversation_token=tokenizer.eos_token,
        reload=True,
        template=args.actor_template,
    )
    if unsupervised_training_enabled:
        unsupervised_train_dataset = get_unsupervised_data(args, tokenizer)
    else:
        unsupervised_train_dataset = None

    # DataLoaders creation:
    data_collator = DataCollatorRLHF(args.max_prompt_seq_len, args.inference_tp_size)
    if args.local_rank == -1:
        prompt_train_sampler = RandomSampler(prompt_train_dataset)
        prompt_eval_sampler = RandomSampler(prompt_eval_dataset)
        ppl_eval_sampler = RandomSampler(ppl_eval_dataset)
        if unsupervised_training_enabled:
            unsupervised_train_sampler = RandomSampler(unsupervised_train_dataset)
    else:
        prompt_train_sampler = DistributedSampler(prompt_train_dataset, seed=args.seed)
        prompt_eval_sampler = DistributedSampler(prompt_eval_dataset, seed=args.seed)
        ppl_eval_sampler = DistributedSampler(ppl_eval_dataset, seed=args.seed)
        if unsupervised_training_enabled:
            unsupervised_train_sampler = DistributedSampler(
                unsupervised_train_dataset, seed=args.seed
            )
    prompt_train_dataloader = DataLoader(
        prompt_train_dataset,
        collate_fn=data_collator,
        sampler=prompt_train_sampler,
        batch_size=args.per_device_generation_batch_size,
    )

    prompt_eval_dataloader = DataLoader(
        prompt_eval_dataset,
        collate_fn=data_collator,
        sampler=prompt_eval_sampler,
        batch_size=args.per_device_generation_batch_size,
    )

    ppl_eval_dataloader = DataLoader(
        ppl_eval_dataset,
        collate_fn=default_data_collator,
        sampler=ppl_eval_sampler,
        batch_size=args.per_device_eval_batch_size,
    )
    if unsupervised_training_enabled:
        unsupervised_train_dataloader = DataLoader(
            unsupervised_train_dataset,
            collate_fn=default_data_collator,
            sampler=unsupervised_train_sampler,
            batch_size=args.per_device_generation_batch_size,
        )
    else:
        unsupervised_train_dataloader = [None] * len(
            prompt_train_dataloader
        )  # basically a dummy dataloader

    num_update_steps_per_epoch = (
        min(len(prompt_train_dataloader), len(unsupervised_train_dataloader))
        * (args.per_device_generation_batch_size / args.per_device_train_batch_size)
        / args.gradient_accumulation_steps
    )
    num_total_iters = int(args.num_train_epochs * num_update_steps_per_epoch)

    return (
        prompt_train_dataloader,
        prompt_eval_dataloader,
        ppl_eval_dataloader,
        unsupervised_train_dataloader,
        num_total_iters,
    )

def dpo_loss(logits, ref_logits, x, B, T, C, beta=1, loss_clamp_value=None):
    logits, ref_logits = logits.view(B, 2, T, C), ref_logits.view(B, 2, T, C)
    x = x.view(B, 2, T)
    logits_pos, logits_neg = torch.split(logits, 1, dim=1)
    ref_logits_pos, ref_logits_neg = torch.split(ref_logits, 1, dim=1)
    x_pos, x_neg = torch.split(x, 1, dim=1)
    logits_pos, logits_neg = logits_pos.squeeze(1), logits_neg.squeeze(1)
    ref_logits_pos, ref_logits_neg = ref_logits_pos.squeeze(1), ref_logits_neg.squeeze(1)
    x_pos, x_neg = x_pos.squeeze(1), x_neg.squeeze(1)

    logits_pos, logits_neg = torch.log_softmax(logits_pos, dim=-1), torch.log_softmax(logits_neg, dim=-1)
    ref_logits_pos, ref_logits_neg = torch.log_softmax(ref_logits_pos, dim=-1), torch.log_softmax(ref_logits_neg, dim=-1)

    logits_pos = torch.gather(logits_pos[:,:-1], dim=-1, index=x_pos[:,1:].unsqueeze(-1)).squeeze(-1)
    logits_neg = torch.gather(logits_neg[:,:-1], dim=-1, index=x_neg[:,1:].unsqueeze(-1)).squeeze(-1)
    ref_logits_pos = torch.gather(ref_logits_pos[:,:-1], dim=-1, index=x_pos[:,1:].unsqueeze(-1)).squeeze(-1)
    ref_logits_neg = torch.gather(ref_logits_neg[:,:-1], dim=-1, index=x_neg[:,1:].unsqueeze(-1)).squeeze(-1)
        
    pi_logits = logits_pos - logits_neg
    if loss_clamp_value is not None and loss_clamp_value >0:
        pi_logits = torch.clamp(pi_logits, None, loss_clamp_value)
    pi_ref_logits = ref_logits_pos - ref_logits_neg

        # loss = -F.logsigmoid(self.beta * pi_logits) - F.logsigmoid(-self.beta * pi_ref_logits)
    loss = -F.logsigmoid(beta * pi_logits-beta * pi_ref_logits)
        # loss = -F.logsigmoid(self.beta * torch.clamp(pi_logits - pi_ref_logits, None, 5)) # TODO clip on pi_logits
    return loss.mean()

# def evaluation_ppl(trainer, eval_dataloader, device, max_eval_samples=1000):
#     losses = 0
#     num_samples = 0
#     trainer.eval()
#     for step, batch in enumerate(eval_dataloader):
#         batch = to_device(batch, device)
#         with torch.no_grad():
#             outputs = trainer.actor_model(**batch, use_cache=False)

#         loss = outputs.loss
#         losses += loss.float()

#         num_samples += len(batch["input_ids"]) * torch.distributed.get_world_size()
#         if num_samples >= max_eval_samples:
#             break
#     losses = losses / (step + 1)
#     try:
#         losses = get_all_reduce_mean(losses)
#     except:
#         pass
#     try:
#         perplexity = torch.exp(losses).item()
#     except OverflowError:
#         perplexity = float("inf")
#     return perplexity, losses.item()
def evaluation_ppl(model, eval_dataloader, max_eval_samples, device):
    model.eval()
    losses = 0
    num_samples = 0
    for step, (x, attn_mask, label) in enumerate(eval_dataloader):
        # batch = to_device(batch, device)
        x, attn_mask, label = x.to(device), attn_mask.to(device), label.to(device)
        with torch.no_grad():
            # outputs = model(**batch)
            outputs = model(x, attention_mask=attn_mask, labels=label)

        loss = outputs.loss
        losses += loss.float()

        num_samples += len(x) * torch.distributed.get_world_size()
        del outputs
        torch.cuda.empty_cache()
        gc.collect()
        if step >= max_eval_samples:
            break
    losses = losses / (step + 1)
    try:
        losses = get_all_reduce_mean(losses)
    except:
        pass
    try:
        perplexity = torch.exp(losses).item()
    except OverflowError:
        perplexity = float("inf")
    model.train()
    return perplexity, losses.item()


# @torch.no_grad()
def evaluation_reward(
    model,
    ref_model,
    eval_dataloader,
    device,
    args,
    global_step,
    deterministic=False,
    print_answers=False,
    max_eval_samples=1000,
):
    eval_reward = []
    eval_length = []
    eval_kl = []
    eval_entropy = []
    num_samples = 0
    model.eval()
    ref_model.eval()
    
    rank = torch.distributed.get_rank()
    if rank == 0:
        # iterator = tqdm(enumerate(eval_dataloader), total=len(eval_dataloader))
        iterator = enumerate(eval_dataloader)
    else:
        iterator = enumerate(eval_dataloader)
    
    for step, (x, attn_mask) in iterator:
        # batch = to_device(batch_prompt, device)
        # x, attn_mask = to_device(x, device), to_device(attn_mask, device)
        B,_, T = x.size()
        # print_rank_0(f"x size: {x.size()}")
        x, attn_mask = x.reshape(-1, T), attn_mask.reshape(-1, T)
        x, attn_mask = x.to(device), attn_mask.to(device)
        # implement Direct Preference Optimization
        with torch.no_grad():
            logits = model(x, attention_mask=attn_mask).logits
            ref_logits = ref_model(x, attention_mask=attn_mask).logits
        C = logits.size(-1)
        logits, ref_logits = logits.view(B, 2, T, C), ref_logits.view(B, 2, T, C)
        x = x.view(B, 2, T)
        logits_pos, logits_neg = torch.split(logits, 1, dim=1)
        ref_logits_pos, ref_logits_neg = torch.split(ref_logits, 1, dim=1)
        x_pos, x_neg = torch.split(x, 1, dim=1)
        logits_pos, logits_neg = logits_pos.squeeze(1), logits_neg.squeeze(1)
        ref_logits_pos, ref_logits_neg = ref_logits_pos.squeeze(1), ref_logits_neg.squeeze(1)
        x_pos, x_neg = x_pos.squeeze(1), x_neg.squeeze(1)

        logits_pos, logits_neg = torch.log_softmax(logits_pos, dim=-1), torch.log_softmax(logits_neg, dim=-1)
        ref_logits_pos, ref_logits_neg = torch.log_softmax(ref_logits_pos, dim=-1), torch.log_softmax(ref_logits_neg, dim=-1)

        logits_pos = torch.gather(logits_pos[:,:-1], dim=-1, index=x_pos[:,1:].unsqueeze(-1)).squeeze(-1)
        logits_neg = torch.gather(logits_neg[:,:-1], dim=-1, index=x_neg[:,1:].unsqueeze(-1)).squeeze(-1)
        ref_logits_pos = torch.gather(ref_logits_pos[:,:-1], dim=-1, index=x_pos[:,1:].unsqueeze(-1)).squeeze(-1)
        ref_logits_neg = torch.gather(ref_logits_neg[:,:-1], dim=-1, index=x_neg[:,1:].unsqueeze(-1)).squeeze(-1)
        
        pi_logits = logits_pos - logits_neg
        pi_ref_logits = ref_logits_pos - ref_logits_neg
        
        kl = torch.sum(torch.exp(ref_logits_pos) * (ref_logits_pos - pi_logits), dim=-1).mean()
        eval_kl.append(kl.item())
        reward = F.logsigmoid(args.dpo_beta * pi_logits-args.dpo_beta * pi_ref_logits)
        eval_reward.append(reward.mean().item())
        

        num_samples += x.size(0) * torch.distributed.get_world_size()
        
        del x, attn_mask, logits, ref_logits, logits_pos, logits_neg, ref_logits_pos, ref_logits_neg, x_pos, x_neg, pi_logits, pi_ref_logits, reward
        torch.cuda.empty_cache()
        gc.collect()
        if num_samples >= max_eval_samples:
            break

    model.train()
    return (
        np.mean(eval_reward),
        # np.mean(eval_length).astype(int),
        np.mean(eval_kl),
        # np.mean(eval_entropy),
    )


def save_prompts_and_answers(prompts, answers, rewards, global_step, file_path):
    assert len(prompts) == len(answers), "Mismatched lengths!"
    assert file_path.endswith(".json")
    data = [
        {
            "id": i,
            "global_step": global_step,
            "prompt": prompts[i],
            "answer": answers[i],
            "reward": rewards[i],
        }
        for i in range(len(prompts))
    ]
    if not os.path.exists(file_path):
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, ensure_ascii=False)
    else:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        # Determine the next id value
        next_id = data[-1]["id"] + 1 if data else 0

        # Create new entries and append them to the data list
        new_entries = [
            {
                "id": next_id + i,
                "global_step": global_step,
                "prompt": prompts[i],
                "answer": answers[i],
                "reward": rewards[i],
            }
            for i in range(len(prompts))
        ]
        data.extend(new_entries)

        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, ensure_ascii=False)


def save_model(args, model, tokenizer):
    if args.output_dir is not None and args.save_model:
        print_rank_0("saving the final model ...", args.global_rank)
        model = convert_lora_to_linear_layer(model)

        if args.global_rank == 0:
            save_hf_format(model, tokenizer, args)

        if args.actor_zero_stage == 3:
            # For zero stage 3, each gpu only has a part of the model, so we need a special save function
            save_zero_three_model(
                model, args.global_rank, args.output_dir, zero_stage=args.actor_zero_stage
            )


def gather_log_probs(logits, labels):
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    return log_probs_labels.squeeze(-1)


def main():
    args = parse_args()
    args.tensorboard_path = args.output_dir
    
    if 'mamba' in args.model_name_or_path or 'openelm' in args.model_name_or_path:
        args.flash_attn=False

    if args.local_rank == -1:
        device = torch.device(get_accelerator().device_name())
    else:
        get_accelerator().set_device(args.local_rank)
        device = torch.device(get_accelerator().device_name(), args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        deepspeed.init_distributed()

    args.global_rank = torch.distributed.get_rank()

    
    ds_config = get_train_ds_config(
        offload=args.offload,
        dtype=args.dtype,
        stage=args.actor_zero_stage,
        enable_tensorboard=args.enable_tensorboard,
        tb_path=args.tensorboard_path,
        tb_name="step2_model",
    )
    ds_config["train_micro_batch_size_per_gpu"] = args.per_device_train_batch_size
    ds_config["train_batch_size"] = (
        args.per_device_train_batch_size
        * torch.distributed.get_world_size()
        * args.gradient_accumulation_steps
    )
    
    ref_ds_config = get_train_ds_config(
        offload=args.offload,
        dtype=args.dtype,
        stage=args.reference_zero_stage,
        enable_tensorboard=False,
    )
    ref_ds_config["train_micro_batch_size_per_gpu"] = args.per_device_eval_batch_size
    ref_ds_config["train_batch_size"] = (
        args.per_device_eval_batch_size
        * torch.distributed.get_world_size()
        * args.gradient_accumulation_steps
    )


    # If passed along, set the training seed now.
    set_random_seed(args.seed)
    torch.distributed.barrier()

    if args.global_rank == 0:
        args.world_size = torch.distributed.get_world_size()
        with open(
            os.path.join(args.output_dir, "args.json"), "w", encoding="utf-8"
        ) as f:
            for key, value in args.__dict__.items():
                json.dump({key: value}, f, ensure_ascii=False)
                f.write("\n")
        # save_code(args.output_dir)

        print(f"Tensorboard logs going to: {args.tensorboard_path}")
        writer = SummaryWriter(f"{args.tensorboard_path}")

    # load tokenizer
    tokenizer = load_hf_tokenizer(
        args.tokenizer_path,
        fast_tokenizer=True,
        add_special_tokens=None,
    )
    print_rank_0(
        f"[Actor Tokenizer] BOS token: {tokenizer.bos_token} EOS token: {tokenizer.eos_token} PAD token: {tokenizer.pad_token}",
        args.global_rank,
    )
    # assert (
    #     tokenizer.pad_token is not None and tokenizer.pad_token != tokenizer.eos_token
    # )
    
    assert tokenizer.pad_token is not None

    from dpo_utils.dataset import RLHFDataset, SFTDataset
    train_dataset = RLHFDataset(data_path=args.data_path[0], block_size=args.max_seq_len, split='train', tokenizer=tokenizer)
    eval_dataset = RLHFDataset(data_path=args.data_path[0], block_size=args.max_seq_len, split='test', tokenizer=tokenizer)
    ppl_eval_dataset = SFTDataset(data_path=args.data_path[0], block_size=args.max_seq_len, split='test', tokenizer=tokenizer)
    
    print_rank_0(f"[Actor Dataset] Train dataset size: {len(train_dataset)}, Eval dataset size: {len(eval_dataset)}, PPL eval dataset size: {len(ppl_eval_dataset)}")
    
    if args.local_rank == -1:
        prompt_train_sampler = RandomSampler(train_dataset)
        prompt_eval_sampler = RandomSampler(eval_dataset)
        ppl_eval_sampler = RandomSampler(ppl_eval_dataset)
    else:
        prompt_train_sampler = DistributedSampler(train_dataset, seed=args.seed)
        prompt_eval_sampler = DistributedSampler(eval_dataset, seed=args.seed)
        ppl_eval_sampler = DistributedSampler(ppl_eval_dataset, seed=args.seed)
    train_dataloader = DataLoader(
        train_dataset,
        # collate_fn=data_collator,
        sampler=prompt_train_sampler,
        batch_size=args.per_device_train_batch_size,
    )

    prompt_eval_dataloader = DataLoader(
        eval_dataset,
        # collate_fn=data_collator,
        sampler=prompt_eval_sampler,
        batch_size=args.per_device_train_batch_size,
    )
    ppl_eval_dataloader = DataLoader(
        ppl_eval_dataset,
        # collate_fn=default_data_collator,
        sampler=ppl_eval_sampler,
        batch_size=args.per_device_eval_batch_size,
    )
    
    if args.dtype == "bf16":
        dtype = torch.bfloat16
    elif args.dtype == "fp16":
        dtype = torch.float16
    elif args.dtype == "fp32":
        dtype = torch.float32
    model = create_hf_model(
        AutoModelForCausalLM,
        args.model_name_or_path,
        tokenizer,
        ds_config,
        dropout=args.dropout,
        flash_attn=args.flash_attn,
        torch_dtype=dtype,
    )
    
    ref_model = create_hf_model(
        AutoModelForCausalLM,
        args.model_name_or_path,
        tokenizer,
        ref_ds_config,
        dropout=args.dropout,
        flash_attn=args.flash_attn,
        torch_dtype=dtype,
    )
    
    if args.optimizer == "adamw":
        AdamOptimizer = DeepSpeedCPUAdam if args.offload else FusedAdam
        optimizer_grouped_parameters = get_optimizer_grouped_parameters(
            model, args.weight_decay, args.actor_lora_learning_rate
        )
        optimizer = AdamOptimizer(
            optimizer_grouped_parameters, lr=args.learning_rate, betas=(0.9, 0.95)
        )
    elif args.optimizer == "adamw_pma":
        optimizer = AdamW_PMA(model.parameters(), lr=args.learning_rate,
              betas=(0.9, 0.95),
              weight_decay=args.weight_decay,
              accumulate_steps=args.gradient_accumulation_steps
        )
        args.gradient_accumulation_steps = 1
    elif args.optimizer == "lion":
        # import Lion here
        from lion_pytorch import Lion
        optimizer = Lion(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    elif args.optimizer == "lion_pma":
        optimizer = Lion_PMA(model.parameters(), lr=args.learning_rate,
              betas=(0.9, 0.95),
              weight_decay=args.weight_decay,
              accumulate_steps=args.gradient_accumulation_steps
        )
        args.gradient_accumulation_steps = 1
    
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.num_train_epochs * num_update_steps_per_epoch,
    )
    
    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        config=ds_config,
        lr_scheduler=lr_scheduler,
        dist_init_required=True,
    )
    ref_model, *_ = deepspeed.initialize(
        model=ref_model,
        config = ref_ds_config,
    )
    
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        
    
    # Train!
    print_rank_0("***** Running training *****", args.global_rank)
    print_rank_0(
        f"***** Evaluating perplexity, Epoch {1}/{args.num_train_epochs} *****",
        args.global_rank,
    )
    perplexity, _ = evaluation_ppl(
        model, ppl_eval_dataloader, args.eval_samples, device
    )
    eval_reward, eval_kl = evaluation_reward(
        model,
        ref_model,
        prompt_eval_dataloader,
        device,
        args,
        0,
        deterministic=False,
        print_answers=True,
        max_eval_samples=args.eval_samples,
    )
    print_rank_0(
        f"eval reward: {eval_reward:.2f} | eval kl: {eval_kl:.2f} | eval ppl: {perplexity:.2f}",
        args.global_rank,
    )
    
    global_step = 0

    gradient_norm = 0
    for epoch in range(args.num_train_epochs):
        print_rank_0(
            f"Beginning of Epoch {epoch+1}/{args.num_train_epochs}, Total Micro Batches {len(train_dataloader)}",
            args.global_rank,
        )
        model.train()
        ref_model.eval()
        mean_loss = 0
        
        # for step, batch in enumerate(train_dataloader):
        for step, (x, attn_mask) in enumerate(train_dataloader):
            global_step += 1
            start = time.time()
            # delete the key-value pair with key "prompt"
            # batch = {k: v for k, v in batch.items() if k != "prompt"}
            # print_rank_0(batch.keys())
            
            # batch = to_device(batch, device)
            B,_, T = x.size()
            x, attn_mask = x.reshape(-1, T), attn_mask.reshape(-1, T)
            # x, attn_mask = to_device(x, device), to_device(attn_mask, device)
            x, attn_mask = x.to(device), attn_mask.to(device)
            
            logits = model(x, attention_mask=attn_mask).logits
            _, _, C = logits.shape
            logits = logits.view(-1,C)
            with torch.no_grad():
                ref_logits = ref_model(x, attention_mask=attn_mask).logits.view(-1,C)
            
            loss = dpo_loss(logits, ref_logits, x, B, T, C, beta=args.dpo_beta)
            
            model.backward(loss)
            gradient_norm_ = model.get_global_grad_norm()
            model.step()
            mean_loss += loss.item()
            end = time.time()
            
            model.zero_grad()
            if gradient_norm_ is not None:
                gradient_norm = gradient_norm_

            if torch.distributed.get_rank() == 0:
                print_throughput(model.module, args, end - start, args.global_rank)
                
                hf_config = model.module.config
                num_layers, hidden_size, vocab_size = get_hf_configs(hf_config)
                gpus_per_model = torch.distributed.get_world_size()
                seq_length = args.max_seq_len
                batch_size = args.per_device_train_batch_size
                checkpoint_activations_factor = 4 if args.gradient_checkpointing else 3
                if args.lora_dim > 0:
                    k = args.lora_dim * 2 / hidden_size
                    checkpoint_activations_factor -= (1 - k)
                train_flops_per_iteration = calculate_flops(
                    checkpoint_activations_factor, batch_size, seq_length, hf_config)

                train_tflops = train_flops_per_iteration / ((end - start) * gpus_per_model *
                                                    (10**12))

            if args.print_loss and torch.distributed.get_rank() == 0 and step % 5 == 0:
                print(
                    f"Epoch {epoch + 1}/{args.num_train_epochs} Step {step}/{len(train_dataloader)}, Rank: {torch.distributed.get_rank()}, loss = {loss:.4f} grad norm = {gradient_norm:.4f}"
                )

            if args.enable_tensorboard and torch.distributed.get_rank() == 0:
                summary_events = [
                    ("Train/loss", loss.item(), model.global_samples),
                    (
                        "Train/gradient_norm",
                        gradient_norm,
                        model.global_samples,
                    ),
                    ("Train/flops", train_tflops, model.global_samples),
                ]
                model.monitor.write_events(summary_events)
            
            if (step + 1) % int(len(train_dataloader) // args.eval_interval) == 0:
                print_rank_0(
                    f"***** Evaluating reward, Epoch {epoch + 1}/{args.num_train_epochs} Step {step}/{len(train_dataloader)} *****",
                    args.global_rank,
                )
                print_machine_info(args.global_rank)
                perplexity, _ = evaluation_ppl(
                    model, ppl_eval_dataloader, args.eval_samples, device
                )
                eval_reward, eval_kl = evaluation_reward(
                    model, ref_model, prompt_eval_dataloader, device, args, step
                )
                print_rank_0(
                    f"eval reward: {eval_reward:.2f} | eval kl: {eval_kl:.2f} | eval ppl: {perplexity:.2f}",
                    args.global_rank,
                )
                model.train()
                
                if args.enable_tensorboard and torch.distributed.get_rank() == 0:
                    writer.add_scalar(
                        "eval/reward", eval_reward, global_step=step
                    )
                    writer.add_scalar("eval/kl", eval_kl, global_step=step)
                    writer.add_scalar("eval/ppl", perplexity, global_step=step)
                    writer.flush()

                # save_model(args, rm_model, tokenizer)
            del logits, ref_logits, loss, x, attn_mask
            torch.cuda.empty_cache()
            gc.collect()

        # print_rank_0(
        #     f"Epoch {epoch+1}/{args.num_train_epochs} with loss {mean_loss/(step+1)}",
        #     args.global_rank,
        # )
        # model.tput_timer.update_epoch_count()
        if args.enable_test_mode:
            break

    # Final
    print_rank_0(f"***** Evaluating at final *****", args.global_rank)
    perplexity, _ = evaluation_ppl(
        model, ppl_eval_dataloader, args.eval_samples, device
    )
    eval_reward, eval_kl = evaluation_reward(
        model,
        ref_model,
        prompt_eval_dataloader,
        device,
        args,
        global_step,
        deterministic=False,
        print_answers=True,
        max_eval_samples=args.eval_samples,
    )
    print_rank_0(
        f"eval reward: {eval_reward:.2f} | eval kl: {eval_kl:.2f} | eval ppl: {perplexity:.2f}",
        args.global_rank,
    )
    if args.enable_tensorboard and torch.distributed.get_rank() == 0:
        writer.add_scalar("eval/reward", eval_reward, global_step=global_step)
        writer.add_scalar("eval/kl", eval_kl, global_step=global_step)
        writer.add_scalar("eval/ppl", perplexity, global_step=global_step)
        writer.flush()

    save_model(args, model, tokenizer)


if __name__ == "__main__":
    main()

