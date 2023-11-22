#!/bin/bash

#SBATCH --partition=ckpt
#SBATCH --account=ark
#SBATCH --job-name=score-comp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a40:2
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/score-comp-%A.log
#SBATCH --error=slurm/score-comp-%A.log
#SBATCH --export=all

module load cuda/11.8.0

set -x

export WANDB_DISABLED=true

model=$1

python eval/superni/run_score_completions.py \
    --do_predict \
    --predict_with_generate \
    --evaluation_strategy "no" \
    --model_name_or_path ${MODEL_DIR}/${model} \
    --max_source_length 1024 \
    --generation_max_length 1152 \
    --max_num_instances_per_task 100 \
    --max_num_instances_per_eval_task 100 \
    --add_task_name False \
    --add_task_definition True \
    --num_pos_examples 0 \
    --num_neg_examples 0 \
    --add_explanation False \
    --tk_instruct False \
    --data_dir data/eval/superni/splits/default \
    --task_dir data/eval/superni/tasks \
    --output_dir results/llama-eval-scores-tulu-format-nopos/${model} \
    --overwrite_output_dir \
    --cache_dir ${TRANSFORMERS_CACHE} \
    --overwrite_cache \
    --per_device_eval_batch_size 4 \
    --use_causal_lm \
    --use_tulu_format \
    --run_name ${model}-score-completion \
    --report_to none
