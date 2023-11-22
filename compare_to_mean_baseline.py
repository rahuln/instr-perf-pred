""" compare performance prediction results to a simple baseline that uses the
    mean performance across all training tasks in a split """

import json
import os

import numpy as np
import pandas as pd
import torch

from aggr_instr_perf_pred_results import main


def load_jsonl(fname):
    """ helper function to load JSONL file """
    with open(fname, 'r') as f:
        lines = [elem.strip() for elem in f.readlines()]
    return list(map(json.loads, lines))


if __name__ == "__main__":

    # format strings for indices file and data file
    fmt = "results/perf_pred/superni/{prompt_format}/random-split-{metric}/" \
          + "{pred_model}/{inst_model}/ep20/seed_{seed}/ep20_bsz16_lr1e-4/" \
          + "indices.pt"
    data_fmt = "data/perf_pred/{prompt_format}/{model}_superni_test_tasks.csv"

    # lists of options for experiment settings
    seeds = list(range(42, 52))
    prompt_format_dir = "defn_tulu_format"
    metrics = ["rougeL", "exact-match"]
    pred_models = ["roberta-base", "roberta-large", "llama-13b"]
    inst_models = ["llama-13b-alpaca", "llama-13b-sharegpt", "tulu-13b",
                   "gpt-3.5-turbo", "gpt-4"]
    inst_models.extend(["llama-7b-sharegpt",
                   "llama-30b-sharegpt", "llama-65b-sharegpt"])

    # aggregate all results
    df_res = main()

    tuples = list()
    columns = ["metric", "pred_model", "inst_model", "baseline_mean",
               "baseline_std", "pred_mean", "pred_std"]

    # iterate through metrics, predictor models, and instruction-tuned models
    for metric in metrics:
        for pred_model in pred_models:
            for inst_model in inst_models:

                # load data file
                data_fname = data_fmt.format(
                                 prompt_format=prompt_format_dir,
                                 model=inst_model.replace("-", "_")
                             )
                if inst_model.startswith("gpt"):
                    data_fname = data_fname.replace("_tulu_format", "")
                data = pd.read_csv(data_fname, index_col=0)

                # get train and test performance values, compute mean baseline
                baseline_rmse = list()
                missing_indices = False
                for seed in seeds:
                    fname = fmt.format(prompt_format=prompt_format_dir,
                                       metric=metric, pred_model=pred_model,
                                       inst_model=inst_model, seed=seed)
                    if inst_model.startswith("gpt"):
                        fname = fname.replace("_tulu_format", "")

                    # if indices file is missing, skip
                    if not os.path.exists(fname):
                        missing_indices = True
                        break

                    # calculate RMSE of mean baseline for this split
                    idxs = torch.load(fname)
                    key = f"predict_{metric.replace('-', '_')}"
                    train_values = data.iloc[idxs["train"]][key].values
                    train_mean = np.mean(train_values)
                    test_values = data.iloc[idxs["test"]][key].values
                    baseline_rmse.append(
                        np.sqrt(np.mean((test_values - train_mean) ** 2))
                    )

                # if indices file missing, continue to next result
                if missing_indices:
                    continue

                if len(baseline_rmse) != len(seeds):
                    print("number of baseline values not matching number of "
                          f"seeds: {(metric, pred_model, inst_model, seed)}")
                    continue

                # calculate mean and std. dev. of baseline RMSE values
                mean_baseline_rmse = np.mean(baseline_rmse)
                std_baseline_rmse = np.std(baseline_rmse)

                # get RMSE values from evaluating  performance predictor
                if inst_model.startswith("gpt"):
                    prompt_format = "defn"
                else:
                    prompt_format = "defn_tulu_format"
                sub_df = df_res[(df_res.prompt_format == prompt_format)
                                & (df_res.metric == metric)
                                & (df_res.pred_model == pred_model)
                                & (df_res.inst_model == inst_model)
                                & (df_res.train_dataset == 'superni')]
                test_rmse = sub_df.test_rmse.values

                if len(test_rmse) != len(seeds):
                    print("number of fine-tuned values not matching number of "
                          f"seeds: {(metric, pred_model, inst_model, seed)}")
                    continue

                # add result to list of results
                mean_rmse = np.mean(test_rmse)
                std_rmse = np.std(test_rmse)
                tuples.append((metric,
                               pred_model,
                               inst_model,
                               mean_baseline_rmse,
                               std_baseline_rmse,
                               mean_rmse,
                               std_rmse))

    # construct data frame of results, print results
    df = pd.DataFrame(tuples, columns=columns)
    print(df.round(1))

