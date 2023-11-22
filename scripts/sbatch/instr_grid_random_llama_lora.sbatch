#!/bin/bash

#SBATCH --partition=ckpt
#SBATCH --account=ark
#SBATCH --job-name=ft-eval-lora
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --time=9:00:00
#SBATCH --array=0-119
#SBATCH --output=slurm/ft-eval-lora-%A-%a.log
#SBATCH --error=slurm/ft-eval-lora-%A-%a.log
#SBATCH --export=all

# arguments: <model> <data_file> <pred_model> <target_col_name>"

module load cuda/11.8.0

export MASTER_PORT=$((12345 + $SLURM_ARRAY_TASK_ID))

# get indices for gradient accumulation steps and learning rate
seed=$((42 + ($SLURM_ARRAY_TASK_ID / 12)))
grad_accum_idx=$((($SLURM_ARRAY_TASK_ID / 4) % 3))
learning_rate_idx=$(($SLURM_ARRAY_TASK_ID % 4))

# get gradient accumulation steps (batch size / 2)
gradient_accum_steps_grid=(8 16 32)
gradient_accumulation_steps=${gradient_accum_steps_grid[$grad_accum_idx]}

# get learning rate
learning_rate_grid=(1e-5 2e-5 5e-5 1e-4)
learning_rate=${learning_rate_grid[$learning_rate_idx]}

# get command-line arguments
model=$1
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
echo "PROMPT FORMAT DIRECTORY: ${prompt_format_dir}"

# construct output directory
batch_size=$gradient_accumulation_steps
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

export CUDA_VISIBLE_DEVICES=0

NUM_GPUS=1
BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=${batch_size}
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training model ${pred_model} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

# Lora training
echo -e "\n=== Running training ===\n"
time python scripts/run_perf_pred_llama.py \
    --model_name_or_path ${MODEL_DIR}/${pred_model} \
    --use_lora \
    --lora_rank 256 \
    --lora_alpha 256 \
    --lora_dropout 0.05 \
    --tokenizer_name ${MODEL_DIR}/${pred_model} \
    --use_slow_tokenizer \
    --train_file ${data_file} \
    --max_seq_length 1024 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate ${learning_rate} \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs 20 \
    --output_dir ${output_dir} \
    --save_merged_lora_model \
    --with_tracking \
    --report_to tensorboard \
    --logging_steps 1 \
    --seed ${seed} \
    --frac_train 0.8 \
    --target_col_name ${target_col_name}

# remove saved model weights to free disk space
echo -e "\n=== Removing saved model checkpoint ===\n"
rm -rf ${output_dir}/best_checkpoint

echo -e "\ndone"
