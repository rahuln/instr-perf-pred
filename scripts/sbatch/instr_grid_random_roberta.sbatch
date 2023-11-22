#!/bin/bash

#SBATCH --partition=ckpt
#SBATCH --account=ark
#SBATCH --job-name=pp-ts-rob
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:rtx6k:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --time=12:00:00
#SBATCH --array=0-119
#SBATCH --output=slurm/pp-ts-rob-%A-%a.log
#SBATCH --error=slurm/pp-ts-rob-%A-%a.log
#SBATCH --export=all

export WANDB_DISABLED=true

# get indices for gradient accumulation steps and learning rate
seed=$((42 + ($SLURM_ARRAY_TASK_ID / 12)))
grad_accum_idx=$((($SLURM_ARRAY_TASK_ID / 4) % 3))
learning_rate_idx=$(($SLURM_ARRAY_TASK_ID % 4))

# get gradient accumulation steps (batch size / 2)
gradient_accum_steps_grid=(4 8 16)
gradient_accumulation_steps=${gradient_accum_steps_grid[$grad_accum_idx]}

# get learning rate
learning_rate_grid=(1e-5 5e-5 1e-4 5e-4)
learning_rate=${learning_rate_grid[$learning_rate_idx]}

# get command-line arguments
model=$1
model_alt=$(echo $model | tr "-" "_")
data_file=$2
pred_model=$3
target_col_name=$4
target_col_name_alt=$(echo $target_col_name | tr "_" "-" | sed -z "s/predict-//g")

# get prompt format based on whether model is an OpenAI model or not
if [[ ${model} == gpt* ]]
then
    prompt_format_dir=defn
else
    prompt_format_dir=defn_tulu_format
fi

# construct output directory
batch_size=$((2 * $gradient_accumulation_steps))
basedir=results/perf_pred/superni/${prompt_format_dir}/random-split-${target_col_name_alt}/${pred_model}/${model}/ep20/seed_${seed}
expdir=ep20_bsz${batch_size}_lr${learning_rate}
output_dir=${basedir}/${expdir}
if [ ! -d $output_dir ]
then
    mkdir -p $output_dir
fi

# check for existing result
if [ -f $output_dir/test_results.json ]
then
    echo "results already exist, exiting..."
    exit
fi

set -x

# run training command
time python scripts/run_perf_pred_roberta.py \
    --model_name ${pred_model} \
    --data_file ${data_file} \
    --max_length 512 \
    --output_dir $output_dir \
    --overwrite_output_dir \
    --logging_dir $output_dir \
    --do_train \
    --do_eval \
    --do_predict \
    --num_train_epochs 20 \
    --learning_rate $learning_rate \
    --lr_scheduler_type constant \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --evaluation_strategy epoch \
    --logging_strategy steps \
    --logging_first_step \
    --logging_steps 10 \
    --save_strategy epoch \
    --save_total_limit 1 \
    --load_best_model_at_end True \
    --metric_for_best_model eval_loss \
    --split_by random \
    --seed $seed \
    --target_col_name ${target_col_name} \
    --max_data_len 119

# delete checkpoints, saved model
if [ -f $output_dir/test_results.json ]
then
    rm -rf $output_dir/pytorch_model.bin
    rm -rf $output_dir/checkpoint*
fi
