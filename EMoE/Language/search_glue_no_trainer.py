# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning a 🤗 Transformers model for sequence classification on GLUE."""
import argparse
import json
import logging
import math
import os
import random
from pathlib import Path

import datasets
import evaluate
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from huggingface_hub import Repository, create_repo
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    BertForSequenceClassification,
    GPT2ForSequenceClassification,
)
from transformers.utils import check_min_version, get_full_repo_name, send_example_telemetry
from transformers.utils.versions import require_version

import numpy as np

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.31.0.dev0")

logger = get_logger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--task_name",
        type=str,
        default=None,
        help="The name of the glue task to train on.",
        choices=list(task_to_keys.keys()),
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--cache_dir", type=str, default='./.cache'
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_length` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=32,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rates",
        type=float,
        default=[2e-5, 3e-5, 5e-5],
        nargs="+",
        help="Initial learning rates (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=10, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default="", help="Where to store the final model.")
    parser.add_argument("--seeds", type=int, default=[0, 1, 2], nargs="+", help="seeds for reproducible training.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.set_defaults(with_tracking=True)
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--ignore_mismatched_sizes",
        action="store_true",
        help="Whether or not to enable to load a pretrained model whose head dimensions are different.",
    )
    
    parser.add_argument(
        "--load_model",
        default=None,)
    
    parser.add_argument('--use_fp16', action='store_true', help='Use fp16 precision.')
    parser.add_argument('--to_MoE', action='store_true')
    # parser.add_argument('--moe_layers', type=int, default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 
    #                                                     12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23
    #                                                     ], nargs='*')
    parser.add_argument('--moe_layers', type=int, default=[10,11], nargs='*')
    
    parser.add_argument('--num_experts', type=int, default=16)
    # parser.add_argument('--add_expert_size', type=float, default=0)
    parser.add_argument('--top_k', type=int, default=4)
    parser.add_argument('--expert_repeat', type=int, default=1)
    parser.add_argument('--add_expert_size', type=int, default=0)
    parser.add_argument('--key_gate', action='store_true')
    parser.add_argument('--gate_type', type=str, default='top', choices=['top', 'cosine_top', 'gated_multi_gate', 'gated_multi_gate_t'])
    parser.add_argument('--one_score', action='store_true')
    parser.add_argument('--random_init_gate', action='store_true')
    parser.add_argument('--normalize_one_score_gate', action='store_true')
    parser.add_argument('--one_score_gate_update_momentum', type=float, default=0.0)
    parser.add_argument('--moe_drop', type=float, default=0.1)
    
    parser.add_argument('--freeze', default=[], nargs='*', type=str, help='freeze part in backbone model') # ['embeddings', 'attention', 'blocks', 'patch_embed', 'cls_token', 'norm', 'pos_embed']
    parser.add_argument('--freeze_layer', default=[], nargs='*', type=int, help='freeze part in backbone model') # ['blocks', 'patch_embed', 'cls_token', 'norm', 'pos_embed']
    parser.add_argument('--tune_moe_layers_only', action='store_true')
    
    
    parser.add_argument('--aux_loss_weight', type=float, default=0.01)
    parser.add_argument('--gate_noise', type=float, default=1.0)
    parser.add_argument('--capacity_factor', type=float, default=1.5)
    parser.add_argument('--save_model', action='store_true')
    
    parser.add_argument('--noise_tuning', type=float, default=0.0)
    
    parser.add_argument('--random_cluster', action='store_true')
    
    # adaptive moe
    parser.add_argument('--max_expert_num', default=8, type=int, help='max number of experts')
    parser.add_argument('--adaptive_experts', action='store_true')
    parser.add_argument('--is_gshard_loss', action='store_true')
    
    
    
    args = parser.parse_args()

    # Sanity checks
    if args.task_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a task name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args

def average_metric(metrics):
    if len(metrics) == 0:
        return 0.0
    else:
        return sum(metrics.values()) / len(metrics)


def main(args, seed=0, lr=3e-5):

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    output_dir = os.path.join(args.output_dir, f"s{seed}_lr{lr}")
    print(output_dir)

    accelerator = (
        Accelerator(log_with=args.report_to, project_dir=output_dir, mixed_precision= "fp16" if args.use_fp16 else None) if args.with_tracking else Accelerator(mixed_precision= "fp16" if args.use_fp16 else None)
    )

    if accelerator.is_local_main_process:   
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.WARN,
            # filename=os.path.join(output_dir, "default.log"),
        )
        
        if 'done' in os.listdir(output_dir):
            print("task already done")
            with open(os.path.join(output_dir, 'done'), 'r') as f:
                best_eval_metric = json.load(f)
                return best_eval_metric, average_metric(best_eval_metric)
    accelerator.wait_for_everyone()
        
    # Make one log on every process with the configuration for debugging.
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    set_seed(seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            create_repo(repo_name, exist_ok=True, token=args.hub_token)
            repo = Repository(output_dir, clone_from=repo_name, token=args.hub_token)

            with open(os.path.join(output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).

    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.

    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)

    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset("./dataset/glue", args.task_name, cache_dir=args.cache_dir)
    else:
        # Loading the dataset from local csv or json file.
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = (args.train_file if args.train_file is not None else args.validation_file).split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=args.cache_dir)
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels
    if args.task_name is not None:
        is_regression = args.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(args.model_name_or_path, 
                                        num_labels=num_labels, 
                                        finetuning_task=args.task_name,
                                        cache_dir=args.cache_dir,)
    if args.load_model and os.path.exists(args.load_model):
        print('load model from', args.load_model)
        tokenizer = torch.load(args.load_model+'/tokenizer.pth')
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, 
                                                use_fast=not args.use_slow_tokenizer,
                                                cache_dir=args.cache_dir,)

    if args.load_model and os.path.exists(args.load_model+'/best_model.pth'):
        model = torch.load(args.load_model+'/best_model.pth')
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            ignore_mismatched_sizes=args.ignore_mismatched_sizes,
            cache_dir=args.cache_dir,
        )
        
    if args.to_MoE:
        from moe_utils import bert_to_MoE, gpt_to_MoE
        if 'bert' in args.model_name_or_path:
            bert_to_MoE(args, model)
        elif 'gpt' in args.model_name_or_path:
            gpt_to_MoE(args, model)
        if args.one_score:
            for name, param in model.named_parameters():
                if 'gate' in name:
                    print('freeze', name)
                    param.requires_grad = False
        # print(model)
    
    if args.freeze or args.freeze_layer or args.tune_moe_layers_only:
        print("====== Freeze ======")
        print("freeze", args.freeze)
        print("freeze_layer", args.freeze_layer)
        for n, p in model.named_parameters():
            n_list = n.split('.')
            freeze_flag = False
            if set(args.freeze).intersection(set(n_list)):
                freeze_flag = True
            if 'gpt' in args.model_name_or_path:
                if 'h' in n_list:
                    temp_layer = int(n_list[n_list.index('h') + 1])
                    if temp_layer in args.freeze_layer:
                        freeze_flag = True
                    if args.tune_moe_layers_only:
                        if temp_layer not in args.moe_layers:
                            freeze_flag = True
                if args.tune_moe_layers_only and ('wte' in n_list or 'wpe' in n_list):
                    freeze_flag = True
            elif 'bert' in args.model_name_or_path:
                if 'layer' in n_list:
                    temp_layer = int(n_list[n_list.index('layer') + 1])
                    if temp_layer in args.freeze_layer:
                        freeze_flag = True
                    if args.tune_moe_layers_only:
                        if temp_layer not in args.moe_layers:
                            freeze_flag = True
                if args.tune_moe_layers_only and 'embeddings' in n_list:
                    freeze_flag = True
            if freeze_flag:
                p.requires_grad = False
                logger.warning(f'freeze: {n}')
            else:
                p.requires_grad = True
                logger.warning(f'****train: {n}')

        
    if args.noise_tuning:
        print("====== Noise Tuning ======")
        for name, param in model.named_parameters():
            if 'classifier' not in name and 'score' not in name:
                model.state_dict()[name][:] += ((torch.rand(param.size())-0.5)*args.noise_tuning*torch.std(param)).to(param.device)
                        
    if args.load_model and os.path.exists(args.load_model+'/best_model_ckpt.pth'):
        print('load model check point from', args.load_model)
        model.load_state_dict(torch.load(args.load_model+'/best_model_ckpt.pth'))
    
    if 'gpt' in args.model_name_or_path:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    # Preprocessing the datasets
    if args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if sorted(label_name_to_id.keys()) == sorted(label_list):
            logger.info(
                f"The configuration of the model provided the following label correspondence: {label_name_to_id}. "
                "Using it!"
            )
            label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {sorted(label_name_to_id.keys())}, dataset labels: {sorted(label_list)}."
                "\nIgnoring the model labels as a result.",
            )
    elif args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif args.task_name is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        # Tokenize the texts
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, padding=padding, max_length=args.max_length, truncation=True)

        if "label" in examples:
            if label_to_id is not None:
                # Map labels to IDs (not necessary for GLUE tasks)
                result["labels"] = [label_to_id[l] for l in examples["label"]]
            else:
                # In all cases, rename the column to labels because the model will expect that.
                result["labels"] = examples["label"]
        return result

    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
            desc="Running tokenizer on dataset",
        )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation_matched" if args.task_name == "mnli" else "validation"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None))

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        # try:
        #     experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        # except AttributeError:
        #     experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"]
        accelerator.init_trackers("glue_no_trainer")

    # Get the metric function
    if args.task_name is not None:
        # metric = evaluate.load("glue", args.task_name, cache_dir=args.cache_dir)
        metric = evaluate.load("./dataset/glue/metrics.py", args.task_name)
    else:
        metric = evaluate.load("accuracy")

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_step

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)
    
    best_metric = 0
    no_improvement = 0
    best_eval_metric = {}
    all_eval_results = {}

    # adaptive begin record
    # if args.to_MoE and args.adaptive_experts:
    #     if 'bert' in args.model_name_or_path:
    #         for i, layer in enumerate(model.bert.encoder.layer):
    #             if i in args.moe_layers:
    #                 layer.mlp.begin_record_routing()
    #     elif 'gpt' in args.model_name_or_path:
    #         for i, layer in enumerate(model.transformer.h):
    #             if i in args.moe_layers:
    #                 layer.mlp.begin_record_routing()

    for epoch in range(starting_epoch, args.num_train_epochs):
        
        model.train()
        if args.with_tracking:
            total_loss = 0
        if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader            
        
        for step, batch in enumerate(active_dataloader):

            # adaptive begin
            if args.to_MoE and args.adaptive_experts and step == len(train_dataloader) // 2:
                if 'bert' in args.model_name_or_path:
                    # print('aaa')
                    for i, layer in enumerate(model.bert.encoder.layer):
                        if i in args.moe_layers:
                            layer.mlp.begin_record_routing()
                elif 'gpt' in args.model_name_or_path:
                    for i, layer in enumerate(model.transformer.h):
                        if i in args.moe_layers:
                            layer.mlp.begin_record_routing()

            outputs = model(**batch)
            loss = outputs.loss

            # print(model)
            
            if args.to_MoE and  args.aux_loss_weight > 0:
                if 'bert' in args.model_name_or_path:
                    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                        for i, layer in enumerate(model.module.bert.encoder.layer):
                            if i in args.moe_layers and getattr(layer.mlp, 'l_aux', None) is not None:
                     
                                loss += layer.mlp.l_aux * args.aux_loss_weight
                    else:
                        for i, layer in enumerate(model.bert.encoder.layer):
                            if i in args.moe_layers and getattr(layer.mlp, 'l_aux', None) is not None:
                     
                                loss += layer.mlp.l_aux * args.aux_loss_weight
                            
                elif 'gpt' in args.model_name_or_path:
                    for i, layer in enumerate(model.transformer.h):
                        if i in args.moe_layers and getattr(layer.mlp, 'l_aux', None) is not None:
    
                            loss += layer.mlp.l_aux * args.aux_loss_weight
            
            # We keep track of the loss at each epoch
            if args.with_tracking:
                total_loss += loss.detach().float()
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1
            
            if args.to_MoE and args.one_score:
                if 'bert' in args.model_name_or_path:
                    for i, layer in enumerate(model.bert.encoder.layer):
                        if i in args.moe_layers:
                            layer.mlp.update_one_score_gate()
                elif 'gpt' in args.model_name_or_path:
                    for i, layer in enumerate(model.transformer.h):
                        if i in args.moe_layers:
                            layer.mlp.update_one_score_gate()
               
            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    temp_output_dir = f"step_{completed_steps }"
                    if output_dir is not None:
                        temp_output_dir = os.path.join(output_dir, temp_output_dir)
                    accelerator.save_state(temp_output_dir)

            # adaptive update
            if args.to_MoE and args.adaptive_experts and step == (len(train_dataloader) * 3) // 4:
                if 'bert' in args.model_name_or_path:
                    # print('aaa')
                    for i, layer in enumerate(model.bert.encoder.layer):
                        if i in args.moe_layers:
                            layer.mlp.adaptive_update_experts()
                elif 'gpt' in args.model_name_or_path:
                    for i, layer in enumerate(model.transformer.h):
                        if i in args.moe_layers:
                            layer.mlp.adaptive_update_experts()

            if completed_steps >= args.max_train_steps:
                break
        
         
        model.eval()
        samples_seen = 0
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()
            predictions, references = accelerator.gather((predictions, batch["labels"]))
            # If we are in a multiprocess environment, the last batch has duplicates
            if accelerator.num_processes > 1:
                if step == len(eval_dataloader) - 1:
                    predictions = predictions[: len(eval_dataloader.dataset) - samples_seen]
                    references = references[: len(eval_dataloader.dataset) - samples_seen]
                else:
                    samples_seen += references.shape[0]
            metric.add_batch(
                predictions=predictions,
                references=references,
            )

        eval_metric = metric.compute()
        all_eval_results[epoch] = eval_metric
        average_result = average_metric(eval_metric)
        logger.info(f"epoch {epoch}: {eval_metric}")
        logger.info(f"epoch {epoch}: {average_result}")

        if args.with_tracking:
            with open(output_dir + "/train.log", "a") as f:
                f.write(f"epoch {epoch}: {total_loss.item() / len(train_dataloader)}\n")
                f.write(f"epoch {epoch}: {eval_metric}\n")
                f.write("\n")
            accelerator.log(
                {
                    "accuracy" if args.task_name is not None else "glue": eval_metric,
                    "train_loss": total_loss.item() / len(train_dataloader),
                    "epoch": epoch,
                    "step": completed_steps,
                },
                step=completed_steps,
            )


        if args.push_to_hub and epoch < args.num_train_epochs - 1:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
            )
            if accelerator.is_main_process:
                tokenizer.save_pretrained(output_dir)
                repo.push_to_hub(
                    commit_message=f"Training in progress epoch {epoch}", blocking=False, auto_lfs_prune=True
                )

        if args.checkpointing_steps == "epoch":
            temp_output_dir = f"epoch_{epoch}"
            if output_dir is not None:
                temp_output_dir = os.path.join(output_dir, temp_output_dir)
            accelerator.save_state(temp_output_dir)
            unwrapped_model = accelerator.unwrap_model(model)

        if average_result > best_metric:
            best_metric = average_result
            best_eval_metric = eval_metric
            no_improvement = 0
            if args.save_model:
                unwrapped_model = accelerator.unwrap_model(model)
                torch.save(unwrapped_model.state_dict(), os.path.join(output_dir, "best_model_ckpt.pth"))
                torch.save(tokenizer, os.path.join(output_dir, "tokenizer.pth"))
        else:
            no_improvement += 1
        
        if no_improvement > 3:
            break
            
    if args.with_tracking:
        accelerator.end_training()

    # if args.output_dir is not None:
    #     accelerator.wait_for_everyone()
    #     unwrapped_model = accelerator.unwrap_model(model)
    #     unwrapped_model.save_pretrained(
    #         args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
    #     )
        
    #     if accelerator.is_main_process:
    #         tokenizer.save_pretrained(args.output_dir)
    #         if args.push_to_hub:
    #             repo.push_to_hub(commit_message="End of training", auto_lfs_prune=True)

    if args.task_name == "mnli":
        # Final evaluation on mismatched validation set
        eval_dataset = processed_datasets["validation_mismatched"]
        eval_dataloader = DataLoader(
            eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
        )
        eval_dataloader = accelerator.prepare(eval_dataloader)

        model.eval()
        for step, batch in enumerate(eval_dataloader):
            outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            metric.add_batch(
                predictions=accelerator.gather(predictions),
                references=accelerator.gather(batch["labels"]),
            )

        eval_metric = metric.compute()
        logger.info(f"mnli-mm: {eval_metric}")

    if output_dir is not None:
        with open(os.path.join(output_dir, "all_results.json"), "w") as f:
            json.dump(all_eval_results, f)
        with open(os.path.join(output_dir, "args.json"), "w") as f:
            json.dump(args.__dict__, f)
        with open(os.path.join(output_dir, "done"), "w") as f:
            json.dump(best_eval_metric, f)

    if args.output_dir is not None:
        all_results = {f"lr{lr}_s{seed}_eval_{k}_best": v for k, v in best_eval_metric.items()}
        with open(os.path.join(args.output_dir, "all_results.json"), "a") as f:
            json.dump(all_results, f)
            f.write("\n")
    
    return best_eval_metric, best_metric


if __name__ == "__main__":
    args = parse_args()
    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    
    if args.task_name in ["mnli", "qqp", "qnli"]:
        args.num_train_epochs = min(args.num_train_epochs, 3)
    
    send_example_telemetry("run_glue_no_trainer", args)
    if args.save_model:
        args.output_dir = os.path.join('./results', args.model_name_or_path+"_save", args.task_name + (f'/{args.output_dir}' if args.output_dir else ''))
    else:
        args.output_dir = os.path.join('./results', args.model_name_or_path, args.task_name + (f'/{args.output_dir}' if args.output_dir else ''))
    
    if args.noise_tuning:
        args.output_dir = args.output_dir + f'/noise_tuning_{args.noise_tuning}'
    
    if args.to_MoE:
        if not args.key_gate:
            args.output_dir = args.output_dir + f'/learn_gate_random_{args.random_init_gate}'
        if args.expert_repeat>1:
            args.output_dir = args.output_dir + f"_repeat{args.expert_repeat}"
    if args.freeze or args.freeze_layer:
        args.output_dir = args.output_dir + f'/freeze_{str(args.freeze_layer)}_freeze_{str(args.freeze)}'
    if args.tune_moe_layers_only:
        args.output_dir = args.output_dir + f'/tune_{str(args.moe_layers)}'
    
    if args.to_MoE:
        args.output_dir = args.output_dir + f'/MoE_{args.moe_layers}_experts_{args.num_experts}_top_k_{args.top_k}_key_gate_{args.key_gate}_{args.gate_type}_aux_{args.aux_loss_weight}_noise_{args.gate_noise}_capacity_{args.capacity_factor}'

    
    results = {}
    all_results = {}
    best_results = 0
    best_lr = 0
    for lr in args.learning_rates:
        results[f'lr{lr}'] = 0
        temp_results = []
        for seed in args.seeds:
            best_eval_metric, best_metric = main(args, seed, lr)
            torch.cuda.empty_cache()
            temp_results.append(best_metric*100)
            all_results[f'lr{lr}_s{seed}'] = best_eval_metric
        results[f'lr{lr}'] = np.mean(temp_results)
        results[f'lr{lr}_std'] = np.std(temp_results)
        if results[f'lr{lr}'] > best_results:
            best_results = results[f'lr{lr}']
            best_results_std = results[f'lr{lr}_std']
            best_lr = lr
    
    if args.output_dir is not None:
        with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
            json.dump(results, f)
            json.dump(all_results, f)
        with open(os.path.join(args.output_dir, f"best_lr_{best_lr}_{best_results}_{best_results_std}.txt"), "w") as f:
            f.write(f'best_lr: {best_lr}')
            f.write(f'best_results: {best_results}')