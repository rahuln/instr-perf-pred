#!/bin/bash

#SBATCH --partition=ckpt
#SBATCH --account=ark
#SBATCH --job-name=llama-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm/sni-eval-%A.log
#SBATCH --error=slurm/sni-eval-%A.log
#SBATCH --export=all

module load cuda/11.8.0

set -x

export WANDB_DISABLED=true

model=$1
num_pos_examples=$2

if [ $num_pos_examples -gt 0 ]
then
    suffix="${num_pos_examples}pos"
else
    suffix="nopos"
fi

time python -m eval.superni.run_eval \
    --data_dir data/eval/superni/splits/default \
    --task_dir data/eval/superni/tasks \
    --max_num_instances_per_task 100 \
    --max_num_instances_per_eval_task 100 \
    --max_source_length 1024 \
    --max_target_length 128 \
    --num_pos_examples ${num_pos_examples} \
    --num_neg_examples 0 \
    --add_task_definition \
    --output_dir results/llama-eval/superni/defn_tulu-format/${model} \
    --model ${MODEL_DIR}/${model} \
    --tokenizer ${MODEL_DIR}/${model} \
    --load_in_8bit \
    --use_tulu_format
