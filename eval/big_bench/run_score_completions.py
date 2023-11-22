#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
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
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.

import logging
import os
import sys
from itertools import chain
import json
from dataclasses import dataclass, field
from typing import List, Optional
import random

import datasets
import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
# from datasets.utils.tqdm_utils import set_progress_bar_enabled
from datasets import load_dataset, load_metric
from tqdm import tqdm

import torch
import transformers
from filelock import FileLock
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    LlamaTokenizer,
    MBart50Tokenizer,
    MBart50TokenizerFast,
    MBartTokenizer,
    MBartTokenizerFast,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.file_utils import is_offline_mode
from transformers.trainer_utils import get_last_checkpoint
from bb_collator import DataCollatorBIGBench
from peft import get_peft_model, LoraConfig, TaskType, PeftModel


# set_progress_bar_enabled(False)
logger = logging.getLogger(__name__)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)

# A list of all multilingual tokenizer which require lang attribute.
MULTILINGUAL_TOKENIZERS = [MBartTokenizer, MBartTokenizerFast, MBart50Tokenizer, MBart50TokenizerFast]


@torch.no_grad()
def score_completions(model, tokenizer, scoring_examples, batch_size=1, aggregation="sum", disable_tqdm=False):
    '''
    Each scoring example is a dict, which contains the following keys:
    - prompt: the prompt to score
    - completions: a list of completions to score
    '''

    # unroll the scoring examples
    unrolled_examples = []
    for scoring_example in scoring_examples:
        prompt = scoring_example["prompt"]
        for completion in scoring_example["completions"]:
            unrolled_examples.append({
                "prompt": prompt,
                "completion": completion
            })
    
    if not disable_tqdm:
        progress = tqdm(total=len(unrolled_examples), desc="Scoring Completions")

    scores = []
    for i in range(0, len(unrolled_examples), batch_size):
        batch_prompts = [example["prompt"] for example in unrolled_examples[i:i+batch_size]]
        batch_examples = [
            (example["prompt"] if example["prompt"][-1] in ["\n", " "] else example["prompt"] + " ")
            + example["completion"] for example in unrolled_examples[i:i+batch_size]
        ]
        tokenized_batch = tokenizer(batch_examples, padding="longest", return_tensors="pt")
        if model.device.type == "cuda":
            tokenized_batch = {
                key: value.cuda() for key, value in tokenized_batch.items()
            }
        outputs = model(**tokenized_batch)

        for example_idx, (prompt, example) in enumerate(zip(batch_prompts, batch_examples)):
            tokenized_prompt = tokenizer(prompt, padding=False, return_tensors="pt").input_ids.squeeze(0)
            tokenized_example = tokenizer(example, padding=False, return_tensors="pt").input_ids.squeeze(0)
            completion_ids = tokenized_example[len(tokenized_prompt):]
            
            # get the logits for the entire example, removing the padding logits
            if tokenizer.padding_side == "right":
                example_logits = outputs.logits[example_idx, :len(tokenized_example), :]
            else:            
                example_logits = outputs.logits[example_idx, -len(tokenized_example):, :]

            # get the logits for the completion portion - note we need to shift the index left by 1 because logits are computed for the next token
            completion_logits = example_logits[len(tokenized_prompt)-1:len(tokenized_example)-1, :]
            completion_log_probs = torch.log_softmax(completion_logits, dim=-1)[range(len(completion_ids)), completion_ids]

            if aggregation == "sum":
                score = completion_log_probs.sum().item()
            elif aggregation == "mean":
                score = completion_log_probs.mean().item()
            elif aggregation == "max":
                score = completion_log_probs.max().item()
            else:
                raise ValueError("Invalid aggregation method: {}".format(aggregation))
            scores.append(score)

        if not disable_tqdm:
            progress.update(len(batch_examples))

    # roll up the scores
    rolled_up_scores = {}
    for unrolled_example, score in zip(unrolled_examples, scores):
        prompt = unrolled_example["prompt"]
        completion = unrolled_example["completion"]
        if prompt not in rolled_up_scores:
            rolled_up_scores[prompt] = {}
        rolled_up_scores[prompt][completion] = score

    return rolled_up_scores


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    include_base_model_in_merge: Optional[bool] = field(
        default=False,
        metadata={"help": "Include base model when merging specified models."},
    )
    merging_weights: Optional[List[str]] = field(
        default=None, metadata={"help": "List of weights for merging models."}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    resize_position_embeddings: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
            "the model's position embeddings."
        },
    )
    use_causal_lm: bool = field(
        default=False,
        metadata={"help": "use causal LM instead of seq2seq model"}
    )
    use_lora: Optional[bool] = field(
        default=False,
        metadata={"help": "Use LoRA instead of full model fine-tuning."}
    )
    load_peft: Optional[str] = field(
        default=None,
        metadata={"help": "Path is to saved PEFT model, not full saved model."}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    lang: str = field(default=None, metadata={"help": "Language id for multilingual model."})
    data_dir: str = field(
        default=None, metadata={"help": "The directory for saving the BIG-bench train/dev/test splits."}
    )
    task_dir: str = field(
        default=None, metadata={"help": "The directory for saving the BIG-bench tasks json files."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_num_instances_per_task: int = field(
        default=None, metadata={"help": "The maximum number of instances we will consider for each training task."}
    )
    max_num_instances_per_eval_task: int = field(
        default=500, metadata={"help": "The maximum number of instances we will consider for each validation/test task."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
            "which is used during ``evaluate`` and ``predict``."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default="", metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )

    forced_bos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": "The token to force as the first generated token after the decoder_start_token_id."
            "Useful for multilingual models like mBART where the first generated token"
            "needs to be the target language token (Usually it is the target language token)"
        },
    )
    add_task_name: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to preappend task name before the task input."}
    )
    add_task_definition: Optional[bool] = field(
        default=True,
        metadata={"help": "whether to preappend task definition before the task input."}
    )
    num_pos_examples: Optional[int] = field(
        default=0,
        metadata={"help": "number of in-context positive examples."}
    )
    num_neg_examples: Optional[int] = field(
        default=0,
        metadata={"help": "number of in-context negative examples."}
    )
    add_explanation: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to add explanation for both the postive examples and negtive examples."}
    )
    tk_instruct: Optional[bool] = field(
        default=False,
        metadata={"help": "tk_instruct will train a model combining all valid instruction encodings. This will overwrite the other settings about instruction encoding."} 
    )
    use_tulu_format: Optional[bool] = field(
        default=False,
        metadata={"help": "Add special tokens for same training format as Tulu paper."}
    )

    def __post_init__(self):
        pass


@dataclass
class BBTrainingArguments(Seq2SeqTrainingArguments):
    denser_evaluation: Optional[bool] = field(
        default=False,
        metadata={"help": "If specifid, the model will do more evaluation at the beginning of training."}
    )
    do_demo: bool = field(default=False, metadata={"help": "Whether to run the model as a demo in the terminal."})


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, BBTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    if data_args.source_prefix is None and model_args.model_name_or_path in [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
    ]:
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is the expected, e.g. with "
            "`--source_prefix 'summarize: ' `"
        )

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # get the absolute path of the bb_dataset.py file
    bb_dataset_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "bb_dataset.py"))

    # Get the BIG-bench dataset
    raw_datasets = load_dataset(
        bb_dataset_file_path, 
        data_dir=data_args.data_dir, 
        task_dir=data_args.task_dir, 
        cache_dir=model_args.cache_dir,
        max_num_instances_per_task=data_args.max_num_instances_per_task,
        max_num_instances_per_eval_task=data_args.max_num_instances_per_eval_task,
    )

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = LlamaTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    if model_args.use_causal_lm:
        auto_model_class = AutoModelForCausalLM
    else:
        auto_model_class = AutoModelForSeq2SeqLM
    model = auto_model_class.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        load_in_8bit=True,
        device_map="auto"
    )

    # load parameters from PEFT method or initialize new LoRA parameters
    if model_args.load_peft is not None:
        model = PeftModel.from_pretrained(model, model_args.load_peft)
    if model_args.use_lora:
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8,
            lora_alpha=32, lora_dropout=0.1
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    model.resize_token_embeddings(len(tokenizer))

    if model.config.decoder_start_token_id is None and isinstance(tokenizer, (MBartTokenizer, MBartTokenizerFast)):
        if isinstance(tokenizer, MBartTokenizer):
            model.config.decoder_start_token_id = tokenizer.lang_code_to_id[data_args.lang]
        else:
            model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids(data_args.lang)

    if model.config.decoder_start_token_id is None and not model_args.use_causal_lm:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    if model_args.use_causal_lm and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if (
        hasattr(model.config, "max_position_embeddings")
        and model.config.max_position_embeddings < data_args.max_source_length
    ):
        if model_args.resize_position_embeddings is None:
            logger.warning(
                f"Increasing the model's number of position embedding vectors from {model.config.max_position_embeddings} "
                f"to {data_args.max_source_length}."
            )
            model.resize_position_embeddings(data_args.max_source_length)
        elif model_args.resize_position_embeddings:
            model.resize_position_embeddings(data_args.max_source_length)
        else:
            raise ValueError(
                f"`--max_source_length` is set to {data_args.max_source_length}, but the model only has {model.config.max_position_embeddings}"
                f" position encodings. Consider either reducing `--max_source_length` to {model.config.max_position_embeddings} or to automatically "
                "resize the model's position encodings by passing `--resize_position_embeddings`."
            )

    if isinstance(tokenizer, tuple(MULTILINGUAL_TOKENIZERS)):
        assert (
            data_args.lang is not None
        ), f"{tokenizer.__class__.__name__} is a multilingual tokenizer which requires --lang argument"

        tokenizer.src_lang = data_args.lang
        tokenizer.tgt_lang = data_args.lang

        # For multilingual translation models like mBART-50 and M2M100 we need to force the target language token
        # as the first generated token. We ask the user to explicitly provide this as --forced_bos_token argument.
        forced_bos_token_id = (
            tokenizer.lang_code_to_id[data_args.forced_bos_token] if data_args.forced_bos_token is not None else None
        )
        model.config.forced_bos_token_id = forced_bos_token_id


    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    if training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

    if training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForBIGBench(
        tokenizer,
        model=None,
        padding="max_length" if data_args.pad_to_max_length else "longest",
        max_source_length=data_args.max_source_length - 7,
        max_target_length=data_args.max_target_length,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
        add_task_name=data_args.add_task_name,
        add_task_definition=data_args.add_task_definition,
        num_pos_examples=data_args.num_pos_examples,
        num_neg_examples=data_args.num_neg_examples,
        add_explanation=data_args.add_explanation,
        tk_instruct=data_args.tk_instruct,
        use_tulu_format=data_args.use_tulu_format,
        text_only=True,
    )
    # we don't want to remove unused columns because we will prepare each batch during training, 
    # and some of the information will aslo be used in evaluation.
    training_args.remove_unused_columns = False 

    max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.max_target_length
    )

    # construct list of scoring examples, each with prompt and list of
    # completions (gold labels)
    scoring_examples = list()
    prompt_to_info = dict()
    for ex in tqdm(raw_datasets["test"], desc="processing examples"):
        formatted = data_collator([ex])
        labels = formatted["labels"]
        label = random.choice(labels)
        scoring_examples.append({
            "prompt" : formatted["inputs"][0].strip(),
            "completions" : [label],
        })
        prompt_to_info[formatted["inputs"][0].strip()] = {
            "task" : ex["Task"],
        }

    # compute scores
    scores = score_completions(model, tokenizer, scoring_examples,
                               aggregation="mean")

    # aggregate scores across examples within each task
    values, counts = dict(), dict()
    for prompt, score_dict in scores.items():
        assert len(score_dict) == 1, "incorrect number of completions"
        task = prompt_to_info[prompt]["task"]
        key = list(score_dict.keys())[0]
        if task not in values:
            values[task] = 0
        if task not in counts:
            counts[task] = 0
        values[task] += score_dict[key]
        counts[task] += 1
    tasks = sorted(list(values.keys()))
    avg_scores = {task : values[task] / counts[task] for task in tasks}

    # save to file
    os.makedirs(training_args.output_dir, exist_ok=True)
    with open(os.path.join(training_args.output_dir, "scores_per_task.jsonl"), "w") as f:
        json.dump(avg_scores, f, indent=4)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
