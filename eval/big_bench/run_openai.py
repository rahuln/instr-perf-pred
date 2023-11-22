import glob
import json
import openai
import tqdm
import os
import random
import time
from typing import Optional
from transformers import HfArgumentParser, GPT2TokenizerFast
from datasets import load_dataset
from bb_collator import DataCollatorForBIGBench
from dataclasses import dataclass, field

openai.api_key=os.environ["OPENAI_KEY"]
openai.organization=os.environ["OPENAI_ORG_ID"]


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    lang: str = field(default=None, metadata={"help": "Language id for multilingual model."})
    data_dir: str = field(
        default=None, metadata={"help": "The directory for saving the NaturalInstructions train/dev/test splits."}
    )
    task_dir: str = field(
        default=None, metadata={"help": "The directory for saving the NaturalInstructions tasks json files."}
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
    add_task_definition: Optional[bool] = field(
        default=True,
        metadata={"help": "whether to preappend task definition before the task input."}
    )

    def __post_init__(self):
        pass


@dataclass
class GPT3Arguments(DataTrainingArguments):
    data_dir: str = field(
        default="data/eval/superni/splits", metadata={"help": "The directory for saving the NaturalInstructions train/dev/test splits."}
    )
    output_dir: str = field(
        default="output/openai/", metadata={"help": "The directory for saving the NaturalInstructions train/dev/test splits."}
    )
    gpt_temperature: float = field(
        default=0, metadata={"help": "the temperature of GPT3."}
    )
    gpt_top_p: float = field(
        default=1, metadata={"help": "the top_p parameter of GPT3."}
    )
    engine: str = field(
        default="gpt-3.5-turbo", metadata={"help": "the openai GPT3 engine to use."}
    )
    sleep_time: float = field(
        default=0.05, metadata={"help": "sleep time between requests."}
    )
    


if __name__ == "__main__":
    random.seed(123)
    parser = HfArgumentParser((GPT3Arguments,))
    args, = parser.parse_args_into_dataclasses()

    # get the absolute path of the bb_dataset.py file
    bb_dataset_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "bb_dataset.py"))

    raw_datasets = load_dataset(
        bb_dataset_file_path, 
        data_dir=args.data_dir, 
        task_dir=args.task_dir, 
        max_num_instances_per_task=args.max_num_instances_per_task,
        max_num_instances_per_eval_task=args.max_num_instances_per_eval_task
    )

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    data_collator = DataCollatorForBIGBench(
        tokenizer,
        model=None,
        padding="max_length" if args.pad_to_max_length else "longest",
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        add_task_definition=args.add_task_definition,
        text_only=True
    )

    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, "run_config.json"), "w") as fout:
        json.dump(args.__dict__, fout)

    existing_requests = {}
    if os.path.exists(os.path.join(args.output_dir, "predicted_examples.jsonl")):
        with open(os.path.join(args.output_dir, "predicted_examples.jsonl")) as fin:
            for line in fin:
                request_info = json.loads(line)
                existing_requests[request_info["gpt_input"]] = request_info["gpt_response"]

    num_tokens = 0
    with open(os.path.join(args.output_dir, "predicted_examples.jsonl"), "w") as fout:
        for example in tqdm.tqdm(raw_datasets["test"]):
            encoded_example = data_collator([example])
            example["gpt_input"] = encoded_example["inputs"][0].strip()
            example["gpt_target"] = encoded_example["labels"][0].strip()
            num_tokens += len(tokenizer.tokenize(encoded_example["inputs"][0].strip()))
            if example["gpt_input"] in existing_requests:
                response = existing_requests[example["gpt_input"]]
            else:
                response = openai.ChatCompletion.create(
                    model=args.engine,
                    messages=[
                        {'role': 'system', 'content': 'You are a helpful assistant.'},
                        {'role' : 'user', 'content' : example["gpt_input"]},
                    ],
                    temperature=args.gpt_temperature,
                    max_tokens=args.max_target_length,
                    top_p=args.gpt_top_p,
                    frequency_penalty=0,
                    presence_penalty=0
                )
                time.sleep(args.sleep_time)
            example["gpt_response"] = response
            # Note: we cut the generated text at the first period, since the GPT3 language model sometimes generates more than one sentences.
            # Our results show that this won't affect the instruct-GPT3 model very much, but will significantly improve the original GPT3 LM.
            example["prediction"] = example["gpt_response"]["choices"][0]["message"]["content"].strip().split(".")[0]
            fout.write(json.dumps(example) + "\n")

    print(f'Total number of tokens: {num_tokens}')
