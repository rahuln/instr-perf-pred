""" script to take computed scores (average loss across instance for each of
    a set of tasks) and merge them with an existing dataset (in CSV file
    format) used for performance prediction """

from argparse import ArgumentParser
import json
import os

import pandas as pd


parser = ArgumentParser()
parser.add_argument('datafile', type=str,
                    help='path to CSV file with performance prediction data')
parser.add_argument('scores_file', type=str,
                    help='path to file containing per-task scores')
parser.add_argument('--outfile', type=str, default=None,
                    help='path to new merged output file')


def main(args):
    """ main function """

    df = pd.read_csv(args.datafile, index_col=0)

    with open(args.scores_file, 'r') as f:
        scores = json.load(f)

    scores = {key.split('_')[0] : value for key, value in scores.items()}
    task_nums = [task.split('_')[0] for task in df.task.tolist()]
    ordered_scores = [scores[task_num] for task_num in task_nums]
    df['predict_avg_loss'] = ordered_scores

    if args.outfile is None:
        args.outfile = args.datafile

    df.to_csv(args.outfile)


if __name__ == '__main__':
    main(parser.parse_args())
