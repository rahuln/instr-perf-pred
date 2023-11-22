#!/bin/bash

data_dir=data/eval/big_bench/splits/default
task_dir=data/eval/big_bench/tasks
output_dir=results/openai-eval
max_num_instances_per_eval_task=100

engine=$1

echo "Using OpenAI engine: ${engine}"

mkdir -p ${output_dir}/big_bench/${engine}/defn \

# use OpenAI API to generate outputs for SuperNI test set instances
python eval/big_bench/run_openai.py \
    --data_dir ${data_dir} \
    --task_dir ${task_dir} \
    --overwrite_cache \
    --max_num_instances_per_task 1 \
    --max_num_instances_per_eval_task ${max_num_instances_per_eval_task} \
    --add_task_definition True \
    --max_source_length 1024 \
    --max_target_length 128 \
    --engine ${engine} \
    --output_dir ${output_dir}/big_bench/${engine}/defn \
    --sleep_time 0.3

# compute metrics using generated outputs
python eval/big_bench/compute_metrics.py \
    --predictions ${output_dir}/big_bench/${engine}/defn/predicted_examples.jsonl \
    --track default \
    --compute_per_task_metrics
