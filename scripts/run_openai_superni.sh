#!/bin/bash

data_dir=data/eval/superni/splits/default
task_dir=data/eval/superni/tasks
output_dir=results/openai-eval
max_num_instances_per_eval_task=100

engine=$1
num_pos_examples=$2

if [ $num_pos_examples -gt 0 ]
then
    suffix="+${num_pos_examples}pos"
else
    suffix=""
fi

echo "Using OpenAI engine: ${engine}"
echo "Using ${num_pos_examples} positive examples"

mkdir -p ${output_dir}/superni/${engine}/defn${suffix} \

# use OpenAI API to generate outputs for SuperNI test set instances
python eval/superni/run_openai.py \
    --data_dir ${data_dir} \
    --task_dir ${task_dir} \
    --overwrite_cache \
    --max_num_instances_per_task 1 \
    --max_num_instances_per_eval_task ${max_num_instances_per_eval_task} \
    --add_task_definition True \
    --num_pos_examples ${num_pos_examples} \
    --num_neg_examples 0 \
    --add_explanation False \
    --max_source_length 1024 \
    --max_target_length 128 \
    --engine ${engine} \
    --output_dir ${output_dir}/superni/${engine}/defn${suffix} \
    --sleep_time 0.3

# compute metrics using generated outputs
python eval/superni/compute_metrics.py \
    --predictions ${output_dir}/superni/${engine}/defn${suffix}/predicted_examples.jsonl \
    --track default \
    --compute_per_category_metrics \
    --compute_per_task_metrics
