import argparse
import logging
import random
import sys
from pathlib import Path
from typing import Iterable

import torch
from torch import optim

from allennlp.common.params import Params
from allennlp.common.tee_logger import TeeLogger
from allennlp.data.dataset_readers import SnliReader
from allennlp.data.instance import Instance
from allennlp.data.iterators.basic_iterator import BasicIterator
from allennlp.data.tokenizers.word_tokenizer import WordTokenizer
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.token_indexers.single_id_token_indexer import (
    SingleIdTokenIndexer
)

from tensorboardX import SummaryWriter

from snli.baseline.model import BaselineModel
from snli.trainers import SupervisedTrainer


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--params', required=True)
    parser.add_argument('-s', '--save', required=True)
    args = parser.parse_args()
    return args


def filter_dataset_by_length(dataset: Iterable[Instance],
                             max_length: int) -> Iterable[Instance]:
    filtered_dataset = []
    for instance in dataset:
        if (len(instance.fields['premise'].tokens) <= max_length
                and len(instance.fields['hypothesis'].tokens) <= max_length):
            filtered_dataset.append(instance)
    num_filtered = len(dataset) - len(filtered_dataset)
    filtered_ratio = num_filtered / len(dataset)
    logger.info(f'Filtered {num_filtered} instances '
                f'({filtered_ratio * 100}%)')
    return filtered_dataset


def main():
    args = parse_args()
    params = Params.from_file(args.params)
    save_dir = Path(args.save)
    save_dir.mkdir(parents=True)

    params.to_file(save_dir / 'params.json')

    train_params, model_params = params.pop('train'), params.pop('model')

    random_seed = train_params.pop_int('random_seed', 2019)
    torch.manual_seed(random_seed)
    random.seed(random_seed)

    log_filename = save_dir / 'stdout.log'
    sys.stdout = TeeLogger(
        filename=log_filename, terminal=sys.stdout,
        file_friendly_terminal_output=False)
    sys.stderr = TeeLogger(
        filename=log_filename, terminal=sys.stderr,
        file_friendly_terminal_output=False)

    tokenizer = WordTokenizer(start_tokens=['<s>'], end_tokens=['</s>'],)
    token_indexer = SingleIdTokenIndexer(lowercase_tokens=True)
    dataset_reader = SnliReader(
        tokenizer=tokenizer, token_indexers={'tokens': token_indexer})

    train_labeled_dataset_path = train_params.pop(
        'train_labeled_dataset_path')
    train_unlabeled_dataset_path = train_params.pop(
        'train_unlabeled_dataset_path', None)
    train_labeled_dataset = dataset_reader.read(train_labeled_dataset_path)
    train_labeled_dataset = filter_dataset_by_length(
        dataset=train_labeled_dataset, max_length=30)
    if train_unlabeled_dataset_path is not None:
        train_unlabeled_dataset = dataset_reader.read(
            train_unlabeled_dataset_path)
        train_unlabeled_dataset = filter_dataset_by_length(
            dataset=train_unlabeled_dataset, max_length=30)
    else:
        train_unlabeled_dataset = []

    valid_dataset = dataset_reader.read(
        train_params.pop('valid_dataset_path'))

    vocab = Vocabulary.from_instances(
        instances=train_labeled_dataset + train_unlabeled_dataset,
        max_vocab_size=train_params.pop_int('max_vocab_size', None))
    vocab.save_to_files(save_dir / 'vocab')

    batch_size = train_params.pop_int('batch_size')
    iterator = BasicIterator(batch_size=batch_size)
    iterator.index_with(vocab)

    model = BaselineModel(params=model_params, vocab=vocab)
    optimizer = optim.Adam(params=model.parameters())
    summary_writer = SummaryWriter(log_dir=save_dir / 'log')

    trainer = SupervisedTrainer(
        model=model,
        optimizer=optimizer,
        iterator=iterator,
        train_dataset=train_labeled_dataset,
        validation_dataset=valid_dataset,
        summary_writer=summary_writer,
        serialization_dir=save_dir,
        num_epochs=train_params.pop('num_epochs', 20),
        iters_per_epoch=len(train_labeled_dataset) // batch_size,
        write_summary_every=100,
        validate_every=2000,
        patience=2,
        clip_grad_max_norm=5,
        cuda_device=train_params.pop_int('cuda_device', 0)
    )
    trainer.train()


if __name__ == '__main__':
    main()
