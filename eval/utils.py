import torch
import tqdm
import json
import time
import asyncio
import os
from peft import PeftModel
from transformers import StoppingCriteria


def encode_with_prompt_completion_format(example, tokenizer, max_seq_length):
    '''
    Here we assume each example has 'prompt' and 'completion' fields.
    We concatenate prompt and completion and tokenize them together because otherwise prompt will be padded/trancated 
    and it doesn't make sense to follow directly with the completion.
    '''
    # if prompt doesn't end with space and completion doesn't start with space, add space
    if not example['prompt'].endswith((' ', '\n', '\t')) and not example['completion'].startswith((' ', '\n', '\t')):
        example_text = example['prompt'] + ' ' + example['completion']
    else:
        example_text = example['prompt'] + example['completion']
    example_text = example_text + tokenizer.eos_token
    tokenized_example = tokenizer(example_text, return_tensors='pt', max_length=max_seq_length, truncation=True)
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()
    tokenized_prompt = tokenizer(example['prompt'], return_tensors='pt', max_length=max_seq_length, truncation=True)
    # mask the prompt part for avoiding loss
    labels[:, :tokenized_prompt.input_ids.shape[1]] = -100
    attention_mask = torch.ones_like(input_ids)
    return {
        'input_ids': input_ids.flatten(),
        'labels': labels.flatten(),
        'attention_mask': attention_mask.flatten(),
    }


@torch.no_grad()
def score_completions(model, tokenizer, scoring_examples, disable_tqdm=False):
    '''
    Each scoring example is a dict, which contains the following keys:
    - prompt: the prompt to score
    - completions: a list of completions to score
    '''
    
    if not disable_tqdm:
        progress = tqdm.tqdm(total=len(scoring_examples), desc="Scoring Completions")

    # unroll the scoring examples
    unrolled_examples = []
    for scoring_example in scoring_examples:
        prompt = scoring_example["prompt"]
        for completion in scoring_example["completions"]:
            unrolled_examples.append({
                "prompt": prompt,
                "completion": completion
            })

    scores = []
    # currently we don't support batching, because we want to directly use the loss returned by the model to score each completion.
    for unrolled_example in unrolled_examples:
        encoded_example = encode_with_prompt_completion_format(unrolled_example, tokenizer, max_seq_length=None)
        # unsqueeze the batch dimension
        for key, value in encoded_example.items():
            encoded_example[key] = value.unsqueeze(0)
        if model.device.type == "cuda":
            encoded_example = {
                key: value.cuda() for key, value in encoded_example.items()
            }
        outputs = model(**encoded_example)
        loss = outputs.loss
        scores.append(-loss.item())
        if not disable_tqdm:
            progress.update(1)

    # roll up the scores
    rolled_up_scores = {}
    for unrolled_example, score in zip(unrolled_examples, scores):
        prompt = unrolled_example["prompt"]
        completion = unrolled_example["completion"]
        if prompt not in rolled_up_scores:
            rolled_up_scores[prompt] = {}
        rolled_up_scores[prompt][completion] = score

    return rolled_up_scores


def load_hf_lm_and_tokenizer(
        model_name_or_path, 
        lora_model_name_or_path=None,
        tokenizer_name_or_path=None, 
        device_map="auto", 
        load_in_8bit=False, 
        load_in_half=False,
        gptq_model=False,
        use_fast_tokenizer=False,
        padding_side="left",
    ):
    
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not tokenizer_name_or_path:
        tokenizer_name_or_path = model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=use_fast_tokenizer)
    # set padding side to left for batch generation
    tokenizer.padding_side = padding_side
    # set pad token to eos token if pad token is not set (as is the case for llama models)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if gptq_model:
        from auto_gptq import AutoGPTQForCausalLM
        model_wrapper = AutoGPTQForCausalLM.from_quantized(
            model_name_or_path, device="cuda:0", use_triton=True
        )
        model = model_wrapper.model  
    elif load_in_8bit:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, 
            device_map=device_map, 
            load_in_8bit=True
        )
    else:
        if device_map:
            model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map=device_map)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
            if torch.cuda.is_available():
                model = model.cuda()
        if load_in_half:
            model = model.half()

    # load LoRA weights, merge with base model
    if lora_model_name_or_path is not None:
        lora_model = PeftModel.from_pretrained(model, lora_model_name_or_path, load_in_8bit=load_in_8bit)
        model = lora_model.base_model.merge_and_unload()
        embedding_size = model.get_input_embeddings().weight.shape[0]
        if len(tokenizer) > embedding_size:
            print(f"The vocabulary size of the tokenizer in the lora model folder contains {len(tokenizer)-embedding_size} more tokens than the base model.")
            print("Resizing the token embeddings of the merged model...")
            model.resize_token_embeddings(len(tokenizer))

    model.eval()
    return model, tokenizer

