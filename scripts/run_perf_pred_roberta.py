""" train pretrained language model as a regression model for predicting the
    performance in ROUGE-L of Tk-Instruct-base on a test task using just the
    task instruction """

from argparse import ArgumentParser
from dataclasses import dataclass, field
import json
import os
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)
from tqdm import tqdm


@dataclass
class CustomArguments:
    """ custom command-line arguments """

    data_file: Optional[str] = field(
        default='tk_instruct_base_test_tasks.csv',
        metadata={'help': 'path to file with training data'},
    )
    input_col_name: Optional[str] = field(
        default='definition',
        metadata={'help': 'name of column for input text'},
    )
    target_col_name: Optional[str] = field(
        default='predict_rougeL',
        metadata={'help': 'name of column for target performance metric'},
    )
    dev_idx: Optional[int] = field(
        default=0,
        metadata={'help': 'index of category to use for dev set'},
    )
    split_by: Optional[str] = field(
        default='category',
        metadata={'help': 'attributed to use for train/dev/test split'},
    )
    model_name: Optional[str] = field(
        default='roberta-large',
        metadata={'help': 'name of Huggingface model to fine-tune'},
    )
    max_length: Optional[int] = field(
        default=512,
        metadata={'help': 'maximum length of input sequence'},
    )
    subfrac_train: Optional[float] = field(
        default=1,
        metadata={'help': 'reduced fraction of training tasks to use'},
    )
    mode: Optional[str] = field(
        default='regression',
        metadata={'help': 'whether to train model for regression or classification'},
    )
    max_data_len: Optional[int] = field(
        default=None,
        metadata={'help': 'artificial maximum data length to force same indices for train/dev/test'},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={'help': 'path to Huggingface cache directory'},
    )


class TaskPerformanceDataset(torch.utils.data.Dataset):
    """ dataset used for regression of task performance from from task
        instruction for SuperNaturalInstructions tasks """

    def __init__(self, data_file, tokenizer, max_length=512, subset='train',
                 dev_idx=0, split_by='category', input_col_name='definition',
                 target_col_name='predict_rougeL', subfrac_train=1,
                 max_data_len=None, mode='regression', seed=42):
        super(TaskPerformanceDataset, self).__init__()

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.input_col_name = input_col_name
        self.target_col_name = target_col_name
        self.mode = mode

        np.random.seed(seed)

        # load data file
        df = pd.read_csv(data_file)

        if split_by == 'category':

            # get list of categories and which ones to use as dev/test sets
            categories = sorted(set(df.category.tolist()))
            test_idx = (dev_idx + 1) % len(categories)
            train_idx = np.array(list(set(np.arange(len(df))) - {dev_idx, test_idx}))
            dev_cat, test_cat = categories[dev_idx], categories[test_idx]

            # split data into train/dev/test sets by task category
            data_train = df[(df.category != dev_cat) & (df.category != test_cat)]
            data_dev = df[df.category == dev_cat]
            data_test = df[df.category == test_cat]

        elif split_by == 'random':

            # randomly split tasks into train/dev/test
            if max_data_len is not None:
                train_idx, eval_idx = \
                    train_test_split(np.arange(max_data_len), train_size=0.8)
                dev_idx, test_idx = train_test_split(eval_idx, train_size=0.5)
                other_idx = np.arange(max_data_len, len(df))
                train_idx = np.concatenate((train_idx, other_idx))
            else:
                train_idx, eval_idx = \
                    train_test_split(np.arange(len(df)), train_size=0.8)
                dev_idx, test_idx = train_test_split(eval_idx, train_size=0.5)

            # get train/dev/test data frames by index
            data_train = df.iloc[train_idx]
            data_dev = df.iloc[dev_idx]
            data_test = df.iloc[test_idx]

        elif split_by == 'task':

            # get list of tasks and which ones to use as dev/test sets
            tasks = sorted(set(df.task.tolist()))
            train_idx, eval_idx = \
                train_test_split(np.arange(len(tasks)), train_size=0.8)
            dev_idx, test_idx = train_test_split(eval_idx, train_size=0.5)

            # restrict indices of training tasks
            if subfrac_train < 1:
                train_size = int(subfrac_train * len(train_idx))
                train_idx = np.random.choice(train_idx, size=train_size,
                                             replace=False)

            train_tasks = set([tasks[idx] for idx in train_idx])
            dev_tasks = set([tasks[idx] for idx in dev_idx])
            test_tasks = set([tasks[idx] for idx in test_idx])

            # split data into train/dev/test sets by task
            data_train = df[df.task.isin(train_tasks)]
            data_dev = df[df.task.isin(dev_tasks)]
            data_test = df[df.task.isin(test_tasks)]

        else:
            raise ValueError(f'unrecognized value for `split_by`: {split_by}')

        # set data file for subset
        if subset == 'train':
            self.data = data_train
            self.idx = train_idx
        elif subset == 'dev':
            self.data = data_dev
            self.idx = dev_idx
        elif subset == 'test':
            self.data = data_test
            self.idx = test_idx

        self.len = len(self.data)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        instruction = row[self.input_col_name]
        inputs = self.tokenizer(instruction, max_length=self.max_length,
                                padding=True, truncation=True)
        dtype = torch.long if self.mode == 'classification' else torch.float
        inputs['label'] = torch.tensor(row[self.target_col_name],
                                       dtype=dtype)
        return inputs


def main(args, training_args):
    """ main script """

    # load tokenizer
    if args.cache_dir is None:
        args.cache_dir = os.environ['TRANSFORMERS_CACHE']
    kwargs = {'cache_dir' : args.cache_dir}
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, **kwargs)

    # construct data subsets
    kwargs = {
        'max_length' : args.max_length,
        'dev_idx' : args.dev_idx,
        'split_by' : args.split_by,
        'input_col_name' : args.input_col_name,
        'target_col_name' : args.target_col_name,
        'subfrac_train' : args.subfrac_train,
        'max_data_len' : args.max_data_len,
        'mode' : args.mode,
        'seed' : training_args.seed,
    }
    train_dataset = TaskPerformanceDataset(args.data_file, tokenizer,
                                           subset='train', **kwargs)
    dev_dataset = TaskPerformanceDataset(args.data_file, tokenizer,
                                         subset='dev', **kwargs)
    test_dataset = TaskPerformanceDataset(args.data_file, tokenizer,
                                          subset='test', **kwargs)

    # load model, indicating that model should perform regression
    num_labels = 2 if args.mode == 'classification' else 1
    problem_type = 'single_label_classification' if args.mode == 'classification' else 'regression'
    kwargs = {
        'cache_dir' : args.cache_dir,
        'num_labels' : num_labels,
        'problem_type' : problem_type,
    }
    model = \
        AutoModelForSequenceClassification.from_pretrained(args.model_name,
                                                           **kwargs)

    def compute_metrics_reg(eval_pred):
        """ regression metrics function """
        y_true, y_pred = eval_pred.label_ids, eval_pred.predictions.flatten()
        metrics = {'mse' : np.mean((y_true - y_pred) ** 2)}
        return metrics

    def compute_metrics_clf(eval_pred):
        """ classification metrics function """
        y_true, y_pred = eval_pred.label_ids, eval_pred.predictions.argmax(axis=1)
        metrics = {
            'accuracy' : accuracy_score(y_true, y_pred),
            'binary_f1' : f1_score(y_true, y_pred, average='binary'),
        }
        return metrics

    # set up trainer
    compute_metrics = compute_metrics_clf if args.mode == 'classification' else compute_metrics_reg
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        data_collator=None,
        compute_metrics=compute_metrics,
    )

    # train
    train_result = trainer.train()

    metrics = train_result.metrics
    trainer.log_metrics('train', metrics)
    trainer.save_metrics('train', metrics)
    trainer.save_state()

    # evaluate on dev set
    dev_results = trainer.predict(dev_dataset, metric_key_prefix='eval')
    metrics = dev_results.metrics
    labels = np.array([elem['label'].item() for elem in dev_dataset])
    metrics['chance_accuracy'] = np.maximum(np.mean(labels), np.mean(1 - labels))
    trainer.log_metrics('eval', metrics)
    trainer.save_metrics('eval', metrics)

    # evaluate on test set
    test_results = trainer.predict(test_dataset, metric_key_prefix='test')
    metrics = test_results.metrics
    labels = np.array([elem['label'].item() for elem in dev_dataset])
    metrics['chance_accuracy'] = np.maximum(np.mean(labels), np.mean(1 - labels))
    trainer.log_metrics('test', metrics)
    trainer.save_metrics('test', metrics)

    # save test set predictions
    test_pred_fname = os.path.join(training_args.output_dir,
                                   'test_predictions.jsonl')
    with open(test_pred_fname, 'w') as f:
        for i, (pred, lab) in enumerate(zip(test_results.predictions.flatten(),
                                        test_results.label_ids)):
            result = {
                'task' : test_dataset.data.iloc[i].task,
                'category' : test_dataset.data.iloc[i].category,
                'prediction' : float(pred),
                'label' : float(lab)
            }
            f.write(json.dumps(result) + '\n')

    trainer.save_model()
    trainer.save_state()

    # save train/dev/test indices
    indices = {
        'train' : train_dataset.idx,
        'dev' : dev_dataset.idx,
        'test' : test_dataset.idx,
    }
    torch.save(indices, os.path.join(training_args.output_dir, 'indices.pt'))

    # save custom command-line arguments
    with open(os.path.join(training_args.output_dir, 'custom_args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)


if __name__ == '__main__':

    # get command-line arguments
    parser = HfArgumentParser((CustomArguments, TrainingArguments))
    args, training_args = parser.parse_args_into_dataclasses()
    main(args, training_args)

