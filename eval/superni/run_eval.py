import json
import tqdm
import os
import random
import torch
import argparse
from datasets import load_dataset
from eval.superni.ni_collator import DataCollatorForNI
from eval.superni.compute_metrics import compute_metrics
from eval.utils import load_hf_lm_and_tokenizer


if __name__ == "__main__":
    random.seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default="/net/nfs.cirrascale/allennlp/yizhongw/natural-instructions/splits/default/",
        help="The directory for saving the NaturalInstructions train/dev/test splits."
    )
    parser.add_argument(
        "--task_dir", 
        type=str, 
        default="/net/nfs.cirrascale/allennlp/yizhongw/natural-instructions/tasks/",
        help="The directory for saving the NaturalInstructions tasks."
    )
    parser.add_argument("--max_num_instances_per_task", type=int, default=1)
    parser.add_argument("--max_num_instances_per_eval_task", type=int, default=10)
    parser.add_argument("--max_source_length", type=int, default=1024)
    parser.add_argument("--max_target_length", type=int, default=1024)
    parser.add_argument("--num_pos_examples", type=int, default=0)
    parser.add_argument("--num_neg_examples", type=int, default=0)
    parser.add_argument("--add_task_definition", action="store_true", help="Whether to add task definition to the input.")
    parser.add_argument("--add_task_name", action="store_true", help="Whether to add task name to the input.")
    parser.add_argument("--add_explanation", action="store_true", help="Whether to add explanation to the input.")
    parser.add_argument("--output_dir", type=str, default="results/superni/", help="The directory for output")
    parser.add_argument("--model", type=str, default="../checkpoints/self_instruct_7B_diverse_templates/")
    parser.add_argument("--tokenizer", type=str, default="../checkpoints/self_instruct_7B_diverse_templates/")
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--load_in_8bit", default=False, action="store_true")
    parser.add_argument("--gptq", action="store_true", help="If given, we're evaluating a 4-bit quantized GPTQ model.")
    parser.add_argument("--use_tulu_format", action="store_true", help="Add user and assistant special tokens to input.")
    parser.add_argument("--output_after_asst", action="store_true", help="Place 'Output:' string after '<|assistant|>' special token.")
    args = parser.parse_args()

    # get the absolute path of the ni_dataset.py file
    ni_dataset_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "ni_dataset.py"))

    raw_datasets = load_dataset(
        ni_dataset_file_path, 
        data_dir=args.data_dir, 
        task_dir=args.task_dir, 
        max_num_instances_per_task=args.max_num_instances_per_task,
        max_num_instances_per_eval_task=args.max_num_instances_per_eval_task
    )

    model, tokenizer = load_hf_lm_and_tokenizer(
        model_name_or_path=args.model, 
        tokenizer_name_or_path=args.tokenizer, 
        load_in_8bit=args.load_in_8bit, 
        load_in_half=True,
        gptq_model=args.gptq
    )

    data_collator = DataCollatorForNI(
        tokenizer,
        model=model,
        padding="longest",
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        add_task_definition=args.add_task_definition,
        num_pos_examples=args.num_pos_examples,
        num_neg_examples=args.num_neg_examples,
        add_explanation=args.add_explanation,
        use_tulu_format=args.use_tulu_format,
        output_after_asst=args.output_after_asst
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # do the evaluation on the test set with batch size

    print("Evaluating on the test set...")
    test_dataset = raw_datasets["test"]
    
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=data_collator,
        drop_last=False,
    )

    # load file with completed batches of examples, crop to nearest batch size
    outfile = os.path.join(args.output_dir, "predictions.jsonl")
    if os.path.exists(outfile):
        with open(outfile, "r") as f:
            entries = list(map(json.loads, list(f.readlines())))
        crop_length = args.eval_batch_size * (len(entries) // args.eval_batch_size)
        entries = entries[:crop_length]
        all_inputs = [entry["input"] for entry in entries]
        predictions = [entry["prediction"] for entry in entries]
    else:
        entries = []
        all_inputs, predictions = [], []

    num_batches = 0
    for batch_idx, batch in enumerate(tqdm.tqdm(test_dataloader, desc="Evaluating on the test set")):

        # if we've already completed this batch, skip this loop
        num_batches += 1
        if len(entries) >= args.eval_batch_size * num_batches:
            continue

        input_ids = batch.input_ids
        attention_mask = batch.attention_mask
        labels = batch.labels

        if model.device.type == "cuda":
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
            labels = labels.cuda()

        # generate the output
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_target_length,
        )

        # # only get the output part
        # outputs = outputs[:, input_ids.shape[1]:]

        # decode the output
        decoded_inputs = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        with open(outfile, "a") as f:
            for idx, (input, output) in enumerate(zip(decoded_inputs, decoded_outputs)):
                all_inputs.append(input)
                assert output.startswith(input)
                predictions.append(output[len(input):])
                # print("-" * 80)
                # print("input: ", input)
                # print("output: ", output[len(input):])

                # get entry from test dataset
                test_entry = test_dataset[batch_idx * args.eval_batch_size + idx]

                # write out input / output entry to file
                entry = {"Task": test_entry["Task"],
                         "Definition": test_entry["Definition"],
                         "Categories": test_entry["Categories"],
                         "Instance": test_entry["Instance"],
                         "input": input,
                         "prediction": output[len(input):]}
                f.write(json.dumps(entry) + "\n")

    # # save the predictions
    # with open(os.path.join(args.output_dir, "predictions.jsonl"), "w") as f:
    #     for input, prediction in zip(all_inputs, predictions):
    #         f.write(json.dumps({"input": input, "prediction": prediction}) + "\n")

    # evaluate the predictions
    print("Evaluating the predictions...")
    references = [ex["Instance"]["output"] for ex in test_dataset]
    metrics = compute_metrics(predictions, references)
    print(metrics)
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f)

