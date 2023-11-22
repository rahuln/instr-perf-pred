# Predicting the Performance of Instruction-tuned Models

This repository contains code to run the experiments described in:

**Third-Party Language Model Performance Prediction from Instruction**<br>
Rahul Nadkarni, Yizhong Wang, Noah A. Smith<br>
_arXiv 2024_

**NOTE:** In order to clone the repo, you will need to have `git-lfs` installed - see [this link](https://git-lfs.com/) for details.

## Setting up environment

The `environment.yml` file contains all of the necessary packages to use this code. We recommend using Anaconda/Miniconda to set up an environment, which you can do with the command:

```
conda-env create -f environment.yml
```

## Downloading instruction datasets

To download the versions of the Super-NaturalInstructions and BIG-bench datasets we use to evaluate instruction-following models, run the following command:

```bash
bash scripts/prepare_data.sh
```

This will download all the necessary files to the `data` directory.

## Reconstructing instruction-tuned models

For the open instruction-tuned models, we use LLaMA-based models trained using the [open-instruct codebase](https://github.com/allenai/open-instruct/) with checkpoints that can be found in the HuggingFace model repository. To download these models, first submit a request for the [LLaMA base models](https://ai.meta.com/blog/large-language-model-llama-meta-ai/). Then download the HuggingFace model diffs for the instruction-tuned models that you want to use; for our experiments, we primarily use `allenai/open-instruct-stanford-alpaca-13b`, `allenai/open-instruct-sharegpt-13b`, and `allenai/tulu-13b`, as well as larger versions of the `sharegpt` models. Finally, you can use the provided `weight_diff.py` script to reconstruct each instruction-tuned model by merging the base model and weight diff as follows:

```Python
python scripts/weight_diff.py make_diff --path_raw {path_to_base_model} --path_tuned {path_to_output_directory} --path_diff {path_to_weight_diff}
```

Once the models have been set up, you should set the `$MODEL_DIR` environment variable to point to the directory where these models are stored. For the closed instruction-tuned models, we use the OpenAI API to run model evaluation -- see below for details.

## Building training datasets

### Main metrics (ROUGE-L and Exact Match)

To build each performance prediction training dataset, we first need to evaluate an instruction-tuned model on an instruction dataset.

For the open LLaMA-based models, there is a `run_eval.py` evaluation script that can be found in the `eval` directory for each dataset, e.g., for Super-NaturalInstructions (SuperNI) the script is `eval/superni/run_eval.py`. Example usage for these scripts can be found in the `llama_eval*.sbatch` Slurm scripts that are set up to run the evaluation in the `scripts/sbatch` directory, e.g., `scripts/sbatch/llama_eval_superni.sbatch`.

For the closed OpenAI models, the corresponding evaluation script is the `run_openai.py` script, e.g., for SuperNI this is `eval/superni/run_openai.py`. Example usage for these scripts can be found in the `run_openai*.sh` BASH scripts in the `scripts` directory, e.g., for SuperNI this is `scripts/run_openai_superni.sh`.

Once you've run the evaluation for an instruction-following model on a specified dataset, you can convert the evaluation results to a `.CSV` file that contains the dataset of {instruction, metric} pairs used to train a performance predictor model. This can be done using the `convert_eval_results_to_perf_pred_dataset.py` script as follows:

```Python
python scripts/convert_eval_results_to_perf_pred_dataset.py {path_to_eval_results_dir} {path_to_output_file} --predictions_fname {path_to_eval_predictions_jsonl_file}
```

### Loss metric

If you want to run performance prediction on the loss metric, you will need to run an additional evaluation to compute the loss scores (this can only be done for SuperNI). To do this, you will run the `eval/superni/run_score_completions.py` script; a Slurm script with an example of how to do this can be found at `scripts/sbatch/score_completions.sbatch`. Once you've computed the per-task average loss scores, you will need to merge them with the dataset file you created above that contains the other per-task evaluation metric values. To do this, you can run the following command:

```Python
python scripts/combine_scores_with_perf_pred_dataset.py {path_to_csv_file} {path_to_per_task_scores_file} --outfile {path_to_output_file}
```

## Fine-tuning performance predictors

Once you've generated the `.CSV` file for the performance prediction training dataset you want to use, you can fine-tune a performance predictor model using several different scripts. For RoBERTa models, you can use the script at `scripts/run_perf_pred_roberta.py`, with an example Slurm script for training RoBERTa performance predictors across a hyperparameter grid located at `scripts/sbatch/instr_grid_random_roberta.sbatch`. For LLaMA-based performance predictors, the corresponding script is `scripts/run_perf_pred_llama.py` with an example Slurm script located at `scripts/sbatch/instr_grid_random_llama_lora.sbatch`. 

## Aggregating results

To see the final performance prediction results, you can use the script `aggr_perf_pred_results.py` which will aggregate all the results into a `pandas` data frame, allowing you to index specific columns to access certain results. To see the simple mean baseline results, use the script `compare_to_mean_baseline.py` to calculate the RMSE values resulting from using the mean baseline and print them out alongside the main performance prediction results.
