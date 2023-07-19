#!/usr/bin/env python
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
"""
Fine-tuning a 🤗 Transformers model on token classification tasks (NER, POS, CHUNKS) relying on the accelerate library
without using a Trainer.
"""

import sys
# sys.path.append("/root/privacy_ner")

import argparse
import json
import logging
import math
import os
import random
from pathlib import Path
import json

import datasets
import torch
from datasets import ClassLabel, load_dataset, load_metric
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from seqeval.metrics import accuracy_score, classification_report
import transformers
from accelerate import Accelerator
from huggingface_hub import Repository
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AdamW,
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
)
from transformers.file_utils import get_full_repo_name
from transformers.utils.versions import require_version
from attack_models.attack_distance_model import knn_attack
import wandb
from seqeval.metrics import accuracy_score, classification_report

logger = logging.getLogger(__name__)
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/token-classification/requirements.txt")

# You should update this to your particular problem to have better documentation of `model_type`
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

def evaluate(model, eval_dataloader, args, config, metric, accelerator, get_labels, compute_metrics):
    model.eval()
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        labels = batch["labels"]
        if not args.pad_to_max_length:  # necessary to pad predictions and labels for being gathered
            predictions = accelerator.pad_across_processes(predictions, dim=1, pad_index=-100)
            labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)

        predictions_gathered = accelerator.gather(predictions)
        labels_gathered = accelerator.gather(labels)
        preds, refs = get_labels(predictions_gathered, labels_gathered)
        metric.add_batch(
            predictions=preds,
            references=refs,
        )  # predictions and preferences are expected to be a nested list of labels, not label_ids

    # eval_metric = metric.compute()
    eval_metric = compute_metrics()
    return eval_metric

def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a text classification task (NER) with accelerate library"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default='conll2003',
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--text_column_name",
        type=str,
        default=None,
        help="The column name of text to input in the file (a csv or JSON file).",
    )
    parser.add_argument(
        "--label_column_name",
        type=str,
        default=None,
        help="The column name of label to input in the file (a csv or JSON file).",
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
         default='./save_models/token/conll2003/phase1',
        # required=True,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
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
        default=16,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=5, help="Total number of training epochs to perform.")
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
        "--num_warmup_steps", type=int, default=1000, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default='./save_models/token/conll2003/phase2', help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=0, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--label_all_tokens",
        action="store_true",
        help="Setting labels of all special tokens to -100 and thus PyTorch will ignore them.",
    )
    parser.add_argument(
        "--return_entity_level_metrics",
        action="store_true",
        help="Indication whether entity level metrics are to be returner.",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="ner",
        choices=["ner", "pos", "chunk"],
        help="The name of the task.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Activate debug mode and run training only with a subset of data.",
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument("--misleading_training", default=1,type=int)
    parser.add_argument("--misleading_weight", default=0.05,type=float)
    parser.add_argument("--target_layer", default=2,type=int)
    parser.add_argument("--use_wandb", default=0,type=int)
    args = parser.parse_args()
    
    # Sanity checks
    # if args.task_name is None and args.train_file is None and args.validation_file is None:
    #     raise ValueError("Need either a task name or a training/validation file.")
    # else:
    #     if args.train_file is not None:
    #         extension = args.train_file.split(".")[-1]
    #         assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
    #     if args.validation_file is not None:
    #         extension = args.validation_file.split(".")[-1]
    #         assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    # if args.push_to_hub:
    #     assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args


def main():
    args = parse_args()

    if args.use_wandb:
        wandb.init(sync_tensorboard=False,
                project="textfusion_phase2_{}_layer{}_other_word".format(args.target_layer, args.dataset_name),
                job_type="CleanRepo",
                config=args,
                name="{}".format(str(args.learning_rate))
                )
    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            repo = Repository(args.output_dir, clone_from=repo_name)
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets for token classification task available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'tokens' or the first column if no column called
    # 'tokens' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.

    tokenizer_name_or_path = args.tokenizer_name if args.tokenizer_name else args.model_name_or_path
    if not tokenizer_name_or_path:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if "gpt2" in args.model_name_or_path or  "roberta" in args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=True, add_prefix_space=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=True)

    from data_utils import build_sentence_dataloader, build_token_dataloader

    padding = "max_length" if args.pad_to_max_length else False

    from data_utils import build_sentence_dataloader, build_token_dataloader
    # dataset_dir = os.path.join('/root/privacy_ner/datasets/', args.dataset_name)
    train_dataloader, eval_dataloader, label_list = build_token_dataloader(args.dataset_name, tokenizer, padding, args.max_length, args.per_device_train_batch_size, args.label_all_tokens, is_test=True)


    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name, num_labels=num_labels)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels)
    else:
        config = CONfFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    

    from models.modeling_bert_textfusion_token import BertForTokenClassification

    config.uncertainty_threshold = 0.1
    config.window_size = 3
    config.fusion_mid_dim = 300
    config.target_layer = [args.target_layer]
    config.phase = 'phase2'
    config.use_mlp = False
    config.misleading_training = args.misleading_training
    model = BertForTokenClassification.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )

    # for i in range(12):
        # model.bert.encoder.token_fusion.uncertainty_classifiers[i].weight = model.classifier.weight
    model.bert.encoder.token_fusion_layer.misleading_layer.mlp.weight.data = model.bert.embeddings.word_embeddings.weight.data.clone().detach()
    
    model.resize_token_embeddings(len(tokenizer))

    # Set the correspondences label/ID inside the model config
    model.config.label2id = {l: i for i, l in enumerate(label_list)}
    model.config.id2label = {i: l for i, l in enumerate(label_list)}

    device = accelerator.device
    model.to(device)
    for name, param in model.named_parameters():
        param.requires_grad = True


    not_allow_optimize_list = ['misleading_layer']
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and not any(allow_name in n for allow_name in not_allow_optimize_list)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and not any(allow_name in n for allow_name in not_allow_optimize_list)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    
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

    # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
    # shorter in multiprocess)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Metrics
    print('load metric')
    metric = load_metric("./metric_seqeval.py")
    print('metirc done')

    def get_labels(predictions, references):
        # Transform predictions and references tensos to numpy arrays
        if device.type == "cpu":
            y_pred = predictions.detach().clone().numpy()
            y_true = references.detach().clone().numpy()
        else:
            y_pred = predictions.detach().cpu().clone().numpy()
            y_true = references.detach().cpu().clone().numpy()

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(pred, gold_label) if l != -100]
            for pred, gold_label in zip(y_pred, y_true)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(pred, gold_label) if l != -100]
            for pred, gold_label in zip(y_pred, y_true)
        ]
        return true_predictions, true_labels

    def compute_metrics():
        results = metric.compute()
        if args.return_entity_level_metrics:
            # Unpack nested dictionaries
            final_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    for n, v in value.items():
                        final_results[f"{key}_{n}"] = v
                else:
                    final_results[key] = value
            return final_results
        else:
            return {
                "precision": results["overall_precision"],
                "recall": results["overall_recall"],
                "f1": results["overall_f1"],
                "accuracy": results["overall_accuracy"],
            }

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataloader)*args.per_device_train_batch_size}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0

    
    best_metric = -1
    best_hit = 100
    misleading_weight = args.misleading_weight

    knn_attack_results = knn_attack(model, tokenizer, eval_dataloader, label_list=label_list, is_token=True)

    target_metric = 'f1'
    candicate_threshold = [0.1,0.2,0.3]

    for epoch in range(args.num_train_epochs):
        if args.use_wandb:
            wandb.log({"epoch": epoch}, step=completed_steps)
        model.train()
        total_loss = 0
        total_misleading_loss = 0
        
        for step, batch in enumerate(train_dataloader):
            if args.use_wandb:
                wandb.log({"Progress": completed_steps}, step=completed_steps)
            config.uncertainty_threshold = random.choice(candicate_threshold)
            config.window_size = random.choice([3])

            model.bert.config.misleading_training = True
            model.bert.config.is_disc = False
            outputs = model(**batch)
            task_loss = outputs.loss['task_loss']
            
            
            optimizer.zero_grad()

            if args.misleading_training:
                misleading_loss = outputs.loss['misleading_loss']
                loss = task_loss + misleading_weight*misleading_loss
                total_misleading_loss += misleading_loss.detach().float()

            total_loss += task_loss.detach().float()

            loss = loss / args.gradient_accumulation_steps

            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            

            # get predictions

            predictions = outputs.logits.argmax(dim=-1)
            labels = batch["labels"]
            if not args.pad_to_max_length:  # necessary to pad predictions and labels for being gathered
                predictions = accelerator.pad_across_processes(predictions, dim=1, pad_index=-100)
                labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)

            predictions_gathered = accelerator.gather(predictions)
            labels_gathered = accelerator.gather(labels)
            preds, refs = get_labels(predictions_gathered, labels_gathered)
            batch_metric = classification_report(preds, refs, output_dict=True)
            batch_f1 = batch_metric['micro avg']['f1-score']
            
            if args.use_wandb:
                for log_type in ['total', 'uncertainty_{}'.format(config.uncertainty_threshold)]:
                    wandb.log({"{}/misleading_loss".format(log_type): misleading_loss.detach().float()}, step=completed_steps)
                    wandb.log({'{}/train_f1'.format(log_type):batch_f1}, step=completed_steps)
                    wandb.log({'{}/train_dis_hit'.format(log_type):outputs.loss['recover_rate']}, step=completed_steps)
                    wandb.log({"{}/task_loss".format(log_type): task_loss.detach().float()}, step=completed_steps)

            progress_bar.update(1)
            progress_bar.set_description('total_loss:{} misleading_loss:{}'.format(total_loss/(step+1), total_misleading_loss/(step+1)))
            completed_steps += 1

            if completed_steps >= args.max_train_steps:
                break
        model.eval()
        
        eval_results = [{'threshould':None, 'metric':None, 'KNN-Attack':None} for i in range(len(candicate_threshold))]
        for i, threshould in enumerate(candicate_threshold):
            config.threshold = threshould
            threshold_type = 'uncertainty_{}'.format(threshould)
            eval_metric = evaluate(model, eval_dataloader, args, config, metric, accelerator, get_labels, compute_metrics)
            logger.info(f"epoch {epoch}, {eval_metric}")
            logger.info(f"KNN-Attack")
            for key,value in knn_attack_results.items():
                logger.info(f"{threshold_type}_{key}: {value}")
            knn_attack_results = knn_attack(model, tokenizer, eval_dataloader, label_list=label_list, is_token=True)
            
            eval_results[i]["threshould"] = threshould
            eval_results[i]["metric"] = eval_metric[target_metric]    
            eval_results[i]["knn_token_hit"] = knn_attack_results
            
            if args.use_wandb:
                
                wandb.log({"dev/{}_{}".format(threshold_type, target_metric): eval_metric[target_metric]}, step=completed_steps)
                for key,value in knn_attack_results.items():
                    wandb.log({"dev/{}_{}".format(threshold_type, key): value}, step=completed_steps)
        
            # if eval_metric[target_metric] >= acceptable_metric and knn_attack_results['token_hit'] <= acceptable_hit:
        new_output_dir = os.path.join(args.output_dir, f'epoch_{epoch}')
        os.makedirs(new_output_dir, exist_ok=True)
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(new_output_dir, save_function=accelerator.save)
        tokenizer.save_pretrained(new_output_dir)
        with open(os.path.join(new_output_dir, "all_results.json"), "w") as f:
            json.dump(eval_results, f)

    
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)
    tokenizer.save_pretrained(args.output_dir)
    with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
        json.dump(eval_results, f)
    
    if args.use_wandb:
        wandb.finish()

    with open('./logs/token_textfusion.txt', 'a') as f:
        f.write('phase2 task: {}, epoch {} f1:{}\n'.format(args.dataset_name, args.model_name_or_path, eval_metric[target_metric]))

    
if __name__ == "__main__":
    main()