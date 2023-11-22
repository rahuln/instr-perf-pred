""" script to aggregate and print performance prediction results """

from argparse import ArgumentParser
from glob import glob
import json
import os

import numpy as np
import pandas as pd


# command-line arguments
parser = ArgumentParser()
parser.add_argument("--basedir", type=str, default=".",
                    help="path to directory containing results directory")


def load_results(dirname, num_seeds=10, num_hyperparams=12, start_seed=42,
    resname="{subset}_results.json", metric="mse", use_prefix=True,
    alt_eval_resname=None):
    """
    load results for all random seeds from a specified directory, choosing
    the test results corresponding to the best eval results across
    hyperparameter settings for each random seed
    """

    seeds = np.arange(start_seed, start_seed + num_seeds)
    metrics = list()
    best_files = list()

    for seed in seeds:

        # load eval metrics
        if alt_eval_resname is not None:
            eval_resname = alt_eval_resname
        else:
            eval_resname = resname.format(subset="eval")
        files = sorted(glob(os.path.join(dirname, f"seed_{seed}", "*",
                                         eval_resname)))
        assert len(files) == num_hyperparams, "incorrect number of results"
        eval_metrics = list()
        key = f"eval_{metric}" if use_prefix else metric
        for fname in files:
            with open(fname, "r") as f:
                res = json.load(f)
            eval_metrics.append(res[key])

        # load test metrics
        files = sorted(glob(os.path.join(dirname, f"seed_{seed}", "*",
                                         resname.format(subset="test"))))
        assert len(files) == num_hyperparams, "incorrect number of results"
        test_metrics = list()
        key = f"test_{metric}" if use_prefix else metric
        for fname in files:
            with open(fname, "r") as f:
                res = json.load(f)
            test_metrics.append(res[key])

        # keep track of test metric corresponding to best eval metric
        metrics.append(test_metrics[np.argmin(eval_metrics)])
        best_files.append(files[np.argmin(eval_metrics)])

    return metrics, best_files


def main(basedir="."):
    """ main function """

    dirname_fmt = "results/perf_pred/{train_dataset}/{prompt}/" \
                  + "random-split-{metric}/{pred_model}/{inst_model}/ep20"
    dirname_fmt = os.path.join(basedir, dirname_fmt)
    columns = ["metric", "pred_model", "inst_model", "test_rmse", "seed",
               "train_dataset", "prompt_format"]

    train_datasets = ["superni", "superni_and_big_bench"]
    prompts = [
        "defn",
        "defn+2pos",
        "defn_tulu_format",
        "defn+2pos_tulu_format",
    ]
    metrics = ["rougeL", "exact-match", "avg-loss"]
    inst_models = ["llama-13b-alpaca", "tulu-13b",
                   "llama-7b-sharegpt", "llama-13b-sharegpt",
                   "llama-30b-sharegpt", "llama-65b-sharegpt",
                   "gpt-3.5-turbo", "gpt-4"]
    inst_models = ["llama-7b-superni", "llama-7b-self-instruct",
                   "llama-7b-code-alpaca",
                   "llama-13b-alpaca", "tulu-13b",
                   "llama-7b-sharegpt", "llama-13b-sharegpt",
                   "llama-30b-sharegpt", "llama-65b-sharegpt",
                   "gpt-3.5-turbo", "gpt-4"]
    pred_models = ["roberta-base", "roberta-large", "llama-13b"]
    
    tuples = list()
    for dataset in train_datasets:
        for prompt in prompts:
            for metric in metrics:
                for inst_model in inst_models:
                    for pred_model in pred_models:
                        dirname = dirname_fmt.format(train_dataset=dataset,
                                                     prompt=prompt,
                                                     metric=metric,
                                                     pred_model=pred_model,
                                                     inst_model=inst_model)
                        if not os.path.exists(dirname):
                            continue
                        try:
                            if pred_model == "llama-13b":
                                best_metric_values, _ = \
                                    load_results(dirname, use_prefix=False,
                                                 alt_eval_resname="best_eval_result.json")
                            else:
                                best_metric_values, _ = load_results(dirname)
                        except AssertionError:
                            print('(', prompt, metric, inst_model, pred_model, ') results incomplete')
                            continue
                        test_rmse = np.sqrt(best_metric_values)
                        for i, val in enumerate(test_rmse):
                            tuples.append((metric, pred_model, inst_model, val,
                                           42 + i, dataset, prompt))

    df = pd.DataFrame(tuples, columns=columns)
    return df


if __name__ == "__main__":
    args = parser.parse_args()
    df = main(args.basedir)

