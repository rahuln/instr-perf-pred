""" script for fine-tuning LLaMA-based model for performance prediction,
    with option of using LoRA for fine-tuning """

#!/usr/bin/env python
# coding=utf-8

import argparse
import json
import logging
import math
import os
import random
import shutil
import string
import datasets
import numpy as np
import torch
from copy import deepcopy
from functools import partial
from glob import glob
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset, Metric
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    LlamaTokenizer,
    SchedulerType,
    DataCollatorWithPadding,
    get_scheduler,
    GPTNeoXTokenizerFast,
    GPT2Tokenizer,
    OPTForCausalLM
)
from peft import LoraConfig, TaskType, get_peft_model

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
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
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="If passed, will use LORA (low-rank parameter-efficient training) to train the model.",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=64,
        help="The rank of lora.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=16,
        help="The alpha parameter of lora.",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,
        help="The dropout rate of lora modules.",
    )
    parser.add_argument(
        "--save_merged_lora_model",
        action="store_true",
        help="If passed, will merge the lora modules and save the entire model.",
    )
    parser.add_argument(
        "--use_flash_attn",
        action="store_true",
        help="If passed, will use flash attention to train the model.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help="The maximum total sequence length (prompt+completion) of each training example.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the eval dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
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
        "--warmup_ratio", type=float, default=0, help="Ratio of total training steps used for warmup."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=None,
        help="Log the training loss and learning rate every logging_steps steps.",
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
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--low_cpu_mem_usage",
        action="store_true",
        help=(
            "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
            "If passed, LLM loading time and RAM consumption will be benefited."
        ),
    )
    parser.add_argument(
        "--frac_train",
        type=float,
        default=0,
        help=(
            "Fraction of examples to use as training data."
        ),
    )
    parser.add_argument(
        "--target_col_name",
        type=str,
        default="predict_rougeL",
        help=(
            "Metric to predict when doing performance prediction.",
        ),
    )
    args = parser.parse_args()

    # Sanity checks
    if args.dataset_name is None and args.train_file is None:
        raise ValueError("Need either a dataset name or a training file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json", "jsonl"], "`train_file` should be a csv/json/jsonl file."
    return args


def main():
    args = parse_args()

    # A hacky way to make llama work with flash attention
    if args.use_flash_attn:
        from llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
        replace_llama_attn_with_flash_attn()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["logging_dir"] = args.output_dir

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    
    accelerator.wait_for_everyone()

    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
        )
    else:
        data_files = {}
        dataset_args = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        raw_datasets = load_dataset(
            "csv",
            data_files=data_files,
            **dataset_args,
        )

    # Load pretrained model and tokenizer
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        raise ValueError(
            "You are instantiating a new config instance from scratch. This is not supported by this script."
        )
    config.problem_type = "regression"
    config.num_labels = 1

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if args.model_name_or_path:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            low_cpu_mem_usage=args.low_cpu_mem_usage,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForSequenceClassification.from_config(config)


    # no default pad token for llama!
    # here we add all special tokens again, because the default ones are not in the special_tokens_map
    if isinstance(tokenizer, LlamaTokenizer):
        num_added_tokens = tokenizer.add_special_tokens({
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
            "pad_token": "<pad>",
        })
        assert num_added_tokens in [0, 1], "LlamaTokenizer should only add one special token - the pad_token, or no tokens if pad token present."
    elif isinstance(tokenizer, GPTNeoXTokenizerFast):
        num_added_tokens = tokenizer.add_special_tokens({
            "pad_token": "<pad>",
        })
        assert num_added_tokens == 1, "GPTNeoXTokenizer should only add one special token - the pad_token."
    elif isinstance(tokenizer, GPT2Tokenizer) and isinstance(model, OPTForCausalLM):
        num_added_tokens = tokenizer.add_special_tokens({'unk_token': '<unk>'})

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    if args.use_lora:
        logger.info("Initializing LORA model...")
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False, 
            r=args.lora_rank, 
            lora_alpha=args.lora_alpha, 
            lora_dropout=args.lora_dropout
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    def encode_function_custom(example, tokenizer, max_seq_length, target_col_name):
        end_tok = '.' if example["definition"][-1] not in string.punctuation else ''
        example_text = f'{example["definition"]}{end_tok}'
        example_text = example_text + ' ' + tokenizer.eos_token
        tokenized_example = tokenizer(example_text, return_tensors='pt',
                                      max_length=max_seq_length,
                                      padding=True, truncation=True)
        input_ids = tokenized_example.input_ids
        label = example[target_col_name]
        attention_mask = torch.ones_like(input_ids)
        return {
            'input_ids': input_ids.flatten(),
            'labels': torch.tensor([label]),
            'attention_mask': attention_mask.flatten(),
        }

    encode_function = partial(
        encode_function_custom,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        target_col_name=args.target_col_name,
    )

    with accelerator.main_process_first():
        lm_datasets = raw_datasets.map(
            encode_function,
            batched=False,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            remove_columns=[name for name in raw_datasets["train"].column_names if name not in ["input_ids", "labels", "attention_mask"]],
            desc="Tokenizing and reformatting instruction data",
        )
        lm_datasets.set_format(type="pt")
        lm_datasets = lm_datasets.filter(lambda example: (example['labels'] != -100).any())

    train_dataset = lm_datasets["train"]

    # split dataset into training, dev, and test
    if args.frac_train > 0:
        dev_dataset = deepcopy(train_dataset)
        test_dataset = deepcopy(train_dataset)

        train_idx, eval_idx = train_test_split(np.arange(len(train_dataset)),
                                               train_size=args.frac_train)
        dev_idx, test_idx = train_test_split(eval_idx, train_size=0.5)
        indices = {'train' : train_idx, 'dev' : dev_idx, 'test' : test_idx}

        train_dataset = train_dataset.select(train_idx)
        dev_dataset = dev_dataset.select(dev_idx)
        test_dataset = test_dataset.select(test_idx)
        test_dataset_info = raw_datasets["train"].select(test_idx)

        torch.save(indices, os.path.join(args.output_dir, 'indices.pt'))

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset, 
        shuffle=True, 
        collate_fn=DataCollatorWithPadding(tokenizer=tokenizer, padding="longest"),
        batch_size=args.per_device_train_batch_size
    )

    dev_dataloader = DataLoader(
        dev_dataset, 
        shuffle=False, 
        collate_fn=DataCollatorWithPadding(tokenizer=tokenizer, padding="longest"),
        batch_size=args.per_device_eval_batch_size
    )

    test_dataloader = DataLoader(
        test_dataset, 
        shuffle=False, 
        collate_fn=DataCollatorWithPadding(tokenizer=tokenizer, padding="longest"),
        batch_size=args.per_device_eval_batch_size
    )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
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
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # Create the learning rate scheduler.
    # Note: the current accelerator.step() calls the .step() of the real scheduler for the `num_processes` times. This is because they assume 
    # the user initialize the scheduler with the entire training set. In the case of data parallel training, each process only
    # sees a subset (1/num_processes) of the training set. So each time the process needs to update the lr multiple times so that the total 
    # number of updates in the end matches the num_training_steps here.
    # Here we need to set the num_training_steps to either using the entire training set (when epochs is specified) or we need to multiply the 
    # num_training_steps by num_processes so that the total number of updates matches the num_training_steps.
    num_training_steps_for_scheduler = args.max_train_steps if overrode_max_train_steps else args.max_train_steps * accelerator.num_processes
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_training_steps=num_training_steps_for_scheduler,
        num_warmup_steps=int(num_training_steps_for_scheduler * args.warmup_ratio),
    )

    # Prepare everything with `accelerator`.
    model, optimizer, train_dataloader, dev_dataloader, test_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, dev_dataloader, test_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
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
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("open_instruct", experiment_config)

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
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)

    # update the progress_bar if load from checkpoint
    progress_bar.update(starting_epoch * num_update_steps_per_epoch)
    completed_steps = starting_epoch * num_update_steps_per_epoch

    # load best RMSE from checkpoint, if it exists
    best_eval_fname = os.path.join(args.output_dir, "best_eval_result.json")
    if os.path.exists(best_eval_fname):
        with open(best_eval_fname, "r") as f:
            best_eval_result = json.load(f)
        best_mse = best_eval_result["mse"]
    else:
        best_mse = float("inf")

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        total_loss = 0
        for step, batch in enumerate(train_dataloader):
            # We need to skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == starting_epoch:
                if resume_step is not None and completed_steps < resume_step:
                    if step % args.gradient_accumulation_steps == 0:
                        progress_bar.update(1)
                        completed_steps += 1
                    continue

            with accelerator.accumulate(model):
                outputs = model(**batch, use_cache=False)
                loss = outputs.loss
                # We keep track of the loss at each logged step
                total_loss += loss.detach().float()
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()       

            # # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

                if args.logging_steps and completed_steps % args.logging_steps == 0:
                    avg_loss = accelerator.gather(total_loss).mean().item() / args.gradient_accumulation_steps / args.logging_steps
                    logger.info(f"  Step: {completed_steps}, LR: {lr_scheduler.get_last_lr()[0]}, Loss: {avg_loss}")
                    if args.with_tracking:
                        accelerator.log(
                            {
                                "learning_rate": lr_scheduler.get_last_lr()[0],
                                "train_loss": avg_loss,
                            },
                            step=completed_steps,
                        )
                    total_loss = 0
                    
                if isinstance(checkpointing_steps, int):
                    if completed_steps % checkpointing_steps == 0:
                        # delete all previous checkpoints to save space
                        dirnames = sorted(glob(os.path.join(args.output_dir, "step_*")))
                        for dirname in dirnames:
                            shutil.rmtree(dirname)

                        output_dir = f"step_{completed_steps}"
                        if args.output_dir is not None:
                            output_dir = os.path.join(args.output_dir, output_dir)
                        accelerator.save_state(output_dir)
                if completed_steps >= args.max_train_steps:
                    break

        # Evaluation loop
        accelerator.wait_for_everyone()
        model.eval()
        if accelerator.is_main_process:
            logger.info("Running evaluation...")
            with torch.no_grad():
                num_examples = 0
                loss = 0
                for eval_step, batch in enumerate(dev_dataloader):
                    outputs = model(**batch, use_cache=False)
                    loss += outputs.loss.item() * len(batch.input_ids)
                    num_examples += len(batch.input_ids)

            eval_mse = loss / num_examples
            logger.info(f"  Step: {completed_steps}, LR: {lr_scheduler.get_last_lr()[0]}, Eval Loss: {eval_mse}")

            # save checkpoint, if new MSE is best
            if eval_mse < best_mse:
                logger.info(f"{eval_mse:.1f} lower than {best_mse:.1f}, saving new best checkpoint")
                best_mse = eval_mse
                with open(os.path.join(args.output_dir, "best_eval_result.json"), "w") as f:
                    json.dump({"mse" : best_mse}, f)

                best_ckpt_dir = os.path.join(args.output_dir, "best_checkpoint")
                os.makedirs(best_ckpt_dir, exist_ok=True)

                tokenizer.save_pretrained(best_ckpt_dir)
                unwrapped_model = accelerator.unwrap_model(model)
                # When doing multi-gpu training, we need to use accelerator.get_state_dict(model) to get the state_dict.
                # Otherwise, sometimes the model will be saved with only part of the parameters.
                # Also, accelerator needs to use the wrapped model to get the state_dict.
                state_dict = accelerator.get_state_dict(model)
                if args.use_lora:
                    # When using lora, the unwrapped model is a PeftModel, which doesn't support the is_main_process 
                    # and has its own save_pretrained function for only saving lora modules.
                    # We have to mannually specify the is_main_process outside the save_pretrained function.
                    if accelerator.is_main_process:
                        unwrapped_model.save_pretrained(best_ckpt_dir, state_dict=state_dict)
                else:
                    unwrapped_model.save_pretrained(
                        best_ckpt_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save, state_dict=state_dict
                    )

        if args.checkpointing_steps == "epoch":
            # delete all previous checkpoints to save space
            dirnames = sorted(glob(os.path.join(args.output_dir, "epoch_*")))
            for dirname in dirnames:
                shutil.rmtree(dirname)

            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)

    if args.with_tracking:
        accelerator.end_training()

    # load parameters from best checkpoint
    best_ckpt_dir = os.path.join(args.output_dir, "best_checkpoint")
    model.load_adapter(best_ckpt_dir, "default")

    # have to manually set the original_module weight
    model.base_model.model.score.original_module.weight = \
        model.base_model.model.score.modules_to_save["default"].weight

    # Evaluation loop for test set
    accelerator.wait_for_everyone()
    model.eval()
    predictions = list()
    labels = list()
    if accelerator.is_main_process:
        logger.info("Running final evaluation...")
        with torch.no_grad():
            num_examples = 0
            loss = 0
            for test_step, batch in enumerate(test_dataloader):
                outputs = model(**batch, use_cache=False)
                loss += outputs.loss.item() * len(batch.input_ids)
                num_examples += len(batch.input_ids)
                predictions.extend(outputs["logits"].detach().cpu().numpy().flatten())
                labels.extend(batch["labels"].cpu().numpy().flatten())

        test_mse = loss / num_examples
        logger.info(f"  Final Test Loss: {test_mse}")

        # save test set metric in file
        with open(os.path.join(args.output_dir, "test_results.json"), "w") as f:
            json.dump({"mse" : test_mse}, f)

        # save all test set predictions to file
        with open(os.path.join(args.output_dir, "test_predictions.jsonl"), "w") as f:
            for info, pred, lab in zip(test_dataset_info, predictions, labels):
                entry = {
                    "task" : info["task"],
                    "category" : info["category"],
                    "prediction" : float(pred),
                    "label" : float(lab),
                }
                f.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    main()
