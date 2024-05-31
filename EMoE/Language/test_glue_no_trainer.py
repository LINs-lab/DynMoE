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
from datasets import load_dataset, load_from_disk
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

from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    LoraConfig,
    PeftType,
    PrefixTuningConfig,
    PromptEncoderConfig,
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
    "cola_ood": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "snli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qnli_ood": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "scitail": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
    "twitter": ("sentence1", "sentence2"),
    "imdb": ("text", None),
    "yelp_polarity": ("text", None),
    "amazon_polarity": ("content", None),
    "flipkart": ("sentence", None),
    "hans": ("premise", "hypothesis"),
    "sick": ("sentence_A", "sentence_B"),
}

test_name_mapping = {
    "imdb": "test",
    "yelp_polarity": "test",
    "amazon_polarity": "test",
    "qqp": "validation",
    "mrpc": "validation",
    'hans': 'validation',
    "sick": 'test',
    "snli": "test",
    "cola_ood": "train",
    "qnli_ood": "train"
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
        "--source_dir", type=str, default=None, help="A csv or a json file containing the training data."
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
        default=64,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=32,
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
    parser.add_argument("--output_dir", type=str, default="test", help="Where to store the final model.")
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
    parser.add_argument('--random_cluster', action='store_true')
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
    parser.add_argument('--gate_type', type=str, default='top', choices=['top', 'cosine_top', 'gated_multi_gate'])
    parser.add_argument('--one_score', action='store_true')
    parser.add_argument('--random_init_gate', action='store_true')
    parser.add_argument('--normalize_one_score_gate', action='store_true')
    parser.add_argument('--one_score_gate_update_momentum', type=float, default=0.0)
    parser.add_argument('--moe_drop', type=float, default=0.1)
    
    
    parser.add_argument('--aux_loss_weight', type=float, default=0.01)
    parser.add_argument('--gate_noise', type=float, default=1.0)
    parser.add_argument('--capacity_factor', type=float, default=1.5)
    parser.add_argument('--save_model', action='store_true')
    
    parser.add_argument('--include_training', action='store_true')
    
    parser.add_argument('--disable_peft', action='store_true')
    
    # adaptive moe
    parser.add_argument('--max_expert_num', default=8, type=int, help='max number of experts')
    parser.add_argument('--adaptive_experts', action='store_true')
    parser.add_argument('--is_gshard_loss', action='store_true')
    
    # parser.add_argument('--normalize', action='store_true', help='normalize the weights of the first linear layer in each FFN')
    # parser.set_defaults(normalize=True)
    
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
    return sum(metrics.values()) / len(metrics)


def main(args, seed, temp_dir):

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    output_dir = temp_dir
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.WARNING,
        # filename=os.path.join(output_dir, "default.log"),
    )
    
    if 'done' not in os.listdir(output_dir):
        print("task training not done")
        return 0, 0

    accelerator = (
        Accelerator(log_with=args.report_to, project_dir=output_dir, mixed_precision= "fp16" if args.use_fp16 else None) if args.with_tracking else Accelerator(mixed_precision= "fp16" if args.use_fp16 else None)
    )

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
        collected_data = False
        if args.task_name in ['qqp', 'mrpc', 'cola', 'sst2', 'stsb', 'mnli', 'qnli', 'rte']:
            raw_datasets = load_dataset("./dataset/glue", args.task_name, cache_dir=args.cache_dir)
        else:
            try:
                raw_datasets = load_dataset(f"./dataset/{args.task_name}", cache_dir=args.cache_dir)
            except:
                collected_data = True
                raw_datasets = load_from_disk(f"./dataset/datasets_self_collected/{args.task_name}")
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
        is_regression = args.task_name == "stsb" or (args.task_name == "sick" and "stsb" in args.source_dir)
        if not is_regression:
            if collected_data:
                label_list = set(raw_datasets["label"])
            else:
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

    # if not args.include_training:
    #     raw_datasets["train"] = raw_datasets["train"][:10]
    
    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(args.model_name_or_path, 
                                        num_labels=num_labels, 
                                        finetuning_task=args.task_name,
                                        cache_dir=args.cache_dir,)
    # if args.load_model and os.path.exists(args.load_model):
    #     print('load model from', args.load_model)
    #     tokenizer = torch.load(args.load_model+'/tokenizer.pth')
    tokenizer = torch.load(temp_dir + '/tokenizer.pth')
    # else:
    #     tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, 
    #                                             use_fast=not args.use_slow_tokenizer,
    #                                             cache_dir=args.cache_dir,)
    print('temp dir: {}'.format(temp_dir))
    if os.path.exists(temp_dir + '/best_model.pth'):
        model = torch.load(temp_dir + '/best_model.pth')
        
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
                if args.one_score:
                    model = model.module
            elif 'gpt' in args.model_name_or_path:
                gpt_to_MoE(args, model)
                if args.one_score:
                    model = model.module
                    
        if not args.disable_peft:
            peft_config = LoraConfig(task_type="SEQ_CLS", inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.1)
            model = get_peft_model(model, peft_config)
            
        print('load model from', temp_dir + '/best_model_ckpt.pth')
        state_dict = torch.load(temp_dir + '/best_model_ckpt.pth')
        
        # print("====model====")
        # print(model)
        with open('./state_dict.txt', 'w') as f:
            f.write(str(state_dict))

        print(model)

        # print([k for k in state_dict.keys()])
        
        model.load_state_dict(state_dict, strict=False)

        with open('./model_state_dict.txt', 'w') as f:
            f.write(str(model.state_dict()))

        
    # if args.load_model and os.path.exists(args.load_model+'/best_model.pth'):
    #     model = torch.load(args.load_model+'/best_model.pth')
    # else:
    #     model = AutoModelForSequenceClassification.from_pretrained(
    #         args.model_name_or_path,
    #         from_tf=bool(".ckpt" in args.model_name_or_path),
    #         config=config,
    #         ignore_mismatched_sizes=args.ignore_mismatched_sizes,
    #         cache_dir=args.cache_dir,
    #     )
    if 'gpt' in args.model_name_or_path:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    # if args.load_model and os.path.exists(args.load_model+'/best_model_ckpt.pth'):
    #     print('load model check point from', args.load_model)
    #     model.load_state_dict(torch.load(args.load_model+'/best_model_ckpt.pth'))
    
    # if 'gpt' in args.model_name_or_path:
    #     tokenizer.pad_token = tokenizer.eos_token
    #     model.config.pad_token_id = model.config.eos_token_id
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
    
    if args.task_name == 'qnli_ood':
        label_to_id = {'entailment': 0, 'not_entailment': 1}

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
                if args.task_name == "sick" and "stsb" in args.source_dir:
                    result["labels"] = examples["relatedness_score"]
                else:
                    result["labels"] = examples["label"]
        return result

    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=raw_datasets["train"].column_names if not collected_data else raw_datasets.column_names,
            desc="Running tokenizer on dataset",
        )
    if not collected_data:
        train_dataset = processed_datasets["train"]
        eval_dataset = processed_datasets[test_name_mapping[args.task_name]]
    else:
        eval_dataset = processed_datasets
        
    for i in range(5):
        print(eval_dataset[i], len(eval_dataset[i]['input_ids']), eval_dataset[i]['labels'])
    
    # Log a few random samples from the training set:
    for index in random.sample(range(len(eval_dataset)), 3):
        logger.info(f"Sample {index} of the eval set: {eval_dataset[index]}.")

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

    if not collected_data:
        train_dataloader = DataLoader(
            train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
        )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    if not collected_data:
        model, train_dataloader, eval_dataloader = accelerator.prepare(model, train_dataloader, eval_dataloader)
    else:
        model, eval_dataloader = accelerator.prepare(model, eval_dataloader)


    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # accelerator.init_trackers("glue_no_trainer", experiment_config)

    # Get the metric function
    if args.task_name == "sick" and "stsb" in args.source_dir:
        metric = evaluate.load("./dataset/glue/metrics.py", "stsb")
    elif args.task_name is not None:
        metric = evaluate.load("./dataset/glue/metrics.py", args.task_name)
    else:
        metric = evaluate.load("accuracy")
    
    # Train!

    logger.info("***** Running test *****")
    # logger.info(f" train set Num examples = {len(train_dataset)}")
    logger.info(f" test set Num examples = {len(eval_dataset)}")


    model.eval()
    samples_seen = 0
    
    if args.include_training:
        for step, batch in enumerate(tqdm(train_dataloader)):
            with torch.no_grad():
                outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()
            predictions, references = accelerator.gather((predictions, batch["labels"]))
            # If we are in a multiprocess environment, the last batch has duplicates
            if accelerator.num_processes > 1:
                if step == len(train_dataloader) - 1:
                    predictions = predictions[: len(train_dataloader.dataset) - samples_seen]
                    references = references[: len(train_dataloader.dataset) - samples_seen]
                else:
                    samples_seen += references.shape[0]
            metric.add_batch(
                predictions=predictions,
                references=references,
            )
    
    for step, batch in enumerate(tqdm(eval_dataloader)):
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

    average_result = average_metric(eval_metric)
    logger.info(f"test all metrics: {eval_metric}")
    logger.info(f"test averaged results: {average_result}")


    # if args.with_tracking:
    #     accelerator.end_training()

    # TODO: update mnli eval later
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
        

    if args.output_dir is not None:
        all_results = {f"test_{k}": v for k, v in eval_metric.items()}
        with open(os.path.join(args.output_dir, f"{args.task_name}_test_results_{'all' if args.include_training else 'test'}.json"), "a") as f:
            json.dump(all_results, f)
            f.write("\n")
    
    return eval_metric, average_result


def str_int_to_list(s):
    split_s = s.split(",")
    for i in range(len(split_s)):
        if i == 0:
            split_s[i] = int(split_s[i][1:] if ']' not in split_s[i][1:] else split_s[i][1:][:-1])
        elif i == len(split_s) - 1:
            split_s[i] = int(split_s[i][:-1])
        else:
            split_s[i] = int(split_s[i])
    return split_s

if __name__ == "__main__":
    args = parse_args()
    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_glue_no_trainer", args)
    load_params = args.source_dir.split("/")[-1]
    load_params = load_params.split("_")
    print(load_params)
    
    if "MoE" in load_params:

        if 'gated' in load_params:
            args.to_MoE = True
            args.moe_layers = str_int_to_list(load_params[load_params.index("MoE") + 1])
            args.max_expert_num = 16
            args.num_experts = 8
            args.adaptive_experts = True
            args.is_gshard_loss = False
            args.key_gate = True if load_params[load_params.index("key") + 2]=="True" else False
            args.gate_type = 'gated_multi_gate' if 't' not in load_params else 'gated_multi_gate_t'
            args.aux_loss_weight = float(load_params[load_params.index("aux") + 1])
            args.gate_noise = float(load_params[load_params.index("noise") + 1])
            args.capacity_factor = float(load_params[load_params.index("capacity") + 1])
        else:
            args.to_MoE = True
            args.moe_layers = str_int_to_list(load_params[load_params.index("MoE") + 1])
            args.num_experts = int(load_params[load_params.index("experts") + 1])
            args.top_k = int(load_params[load_params.index("top") + 2])
            args.key_gate = True if load_params[load_params.index("key") + 2]=="True" else False
            args.gate_type = 'cosine_top' if load_params[load_params.index("gate") + 2] == 'cosine' else 'top'
            args.aux_loss_weight = float(load_params[load_params.index("aux") + 1])
            args.gate_noise = float(load_params[load_params.index("noise") + 1])
            args.capacity_factor = float(load_params[load_params.index("capacity") + 1])
        print("finished loading MoE parameters")
    print(args)

    
        
    args.output_dir = args.source_dir + f"/{args.output_dir}"
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    all_files = os.listdir(args.source_dir)
    
    best_lr = 0
    for file in all_files:
        print(file)
        if ".txt" in file and "best" in file:
            best_lr = float(file.split("_")[2])
            break
    if best_lr == 0:
        print("No best results found")
        exit()
        
    all_results = {}
    average_results = {}
    
    for file in all_files:
        if str(best_lr) in file and not file.endswith(".txt"):
            temp_dir = os.path.join(args.source_dir, file)
            temp_seed = int(file.split("_")[0][1:])
            eval_metric, average_result = main(args, temp_seed, temp_dir)
            torch.cuda.empty_cache()
            all_results[file] = eval_metric
            average_results[file] = average_result * 100
    
    average_results_std = np.std(list(average_results.values()))
    average_results_mean = np.mean(list(average_results.values()))
    
    if args.output_dir is not None:
        with open(os.path.join(args.output_dir, f"{args.task_name}_results_{'all' if args.include_training else 'test'}.json"), "w") as f:
            json.dump(all_results, f)
            json.dump(average_results, f)
        with open(os.path.join(args.output_dir, f"{args.task_name}_{'all' if args.include_training else 'test'}_{average_results_mean}_{average_results_std}.txt"), "w") as f:
            f.write(f'test_results: {average_results_mean}')