""" script to convert results for an instruction-tuned model evaluated on
    SuperNI to a CSV file that can be used for training a performance
    prediction model """

from argparse import ArgumentParser
from collections import defaultdict
from glob import glob
import json
import os

import pandas as pd
from tqdm import tqdm


# command-line arguments
parser = ArgumentParser()
parser.add_argument('resdir', type=str,
                    help='path to model evaluation results directory')
parser.add_argument('savename', type=str,
                    help='filename of location to save evaluation results')
parser.add_argument('--tasks_dir', type=str, default='data/tasks',
                    help='path to directory with SuperNI task info files')
parser.add_argument('--predictions_fname', type=str,
                    default='predictions.jsonl',
                    help='name of file that contains model predictions')


def load_jsonl(fname):
    """ load JSONL file """
    with open(fname, 'r') as f:
        lines = f.readlines()
    return list(map(json.loads, lines))


def main(args):
    """ main function """

    # load evaluation results, get tasks list
    fname = 'predict_results.json'
    with open(os.path.join(args.resdir, fname), 'r') as f:
        metrics = json.load(f)
    examples = [elem.replace('predict_rougeL_for_', '') for elem in metrics
                if elem.startswith('predict_rougeL_for_task')]

    task_nums = set([ex.split('_')[0] for ex in examples])

    # get mapping from task number to task info file
    task_num_to_file = {}
    task_num_to_name = dict()
    for fname in glob(os.path.join(args.tasks_dir, "*.json")):
        task_num = os.path.basename(fname).split("_")[0]
        if task_num not in task_nums:
            continue
        task_name = os.path.basename(fname).replace('.json', '')
        task_num_to_file[task_num] = fname
        task_num_to_name[task_num] = task_name
    tasks = sorted(list(task_num_to_name.values()))

    # load category info for all tasks
    task_to_alt_task = dict()
    task_definitions = dict()
    task_categories = defaultdict(lambda: 'Unknown')
    for task in tqdm(tasks, desc='loading task info'):
        task_num = task.split("_")[0]
        task_file = task_num_to_file[task_num]
        task_name, _ = os.path.splitext(os.path.basename(task_file))
        with open(task_file, 'r') as f:
            task_info = json.load(f)
        task_definitions[task_name] = task_info['Definition'][0]
        if 'Categories' in task_info:
            task_categories[task_name] = task_info['Categories'][0]
        task_to_alt_task[task_name] = task

    # build rows of data frame
    tuples = list()
    for task in tqdm(task_definitions.keys(), desc='getting eval results'):
        alt_task = task_to_alt_task[task]
        tup = (
            task,
            task_definitions[task],
            task_categories[task],
            metrics[f'predict_rougeL_for_{alt_task}'],
            metrics[f'predict_exact_match_for_{alt_task}'],
        )
        tuples.append(tup)

    # construct and save data frame
    columns = ['task', 'definition', 'category', 'predict_rougeL',
               'predict_exact_match']
    df = pd.DataFrame(tuples, columns=columns)
    df.to_csv(args.savename)


if __name__ == '__main__':
    main(parser.parse_args())

