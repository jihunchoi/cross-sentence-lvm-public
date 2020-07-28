import argparse
import logging
import random
import sys
from pathlib import Path
from typing import Iterable

import torch
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer
from torch import optim

from allennlp.common.params import Params
from allennlp.common.tee_logger import TeeLogger
from allennlp.data.dataset_readers import SnliReader
from allennlp.data.iterators.basic_iterator import BasicIterator
from allennlp.data.tokenizers.token import Token
from allennlp.data.tokenizers.word_tokenizer import WordTokenizer
from allennlp.data.vocabulary import Vocabulary, DEFAULT_PADDING_TOKEN

from tensorboardX import SummaryWriter

from snli.deconv_lvm.model import DeconvSNLIModel
from snli.trainers import SeparatedLVMTrainer


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--params', required=True)
    parser.add_argument('-s', '--save', required=True)
    args = parser.parse_args()
    return args


def truncate_or_pad_dataset(dataset: Iterable[Instance],
                            length: int) -> Iterable[Instance]:
    def aux(tokens):
        if len(tokens) < length:
            tokens = (tokens
                      + [Token(DEFAULT_PADDING_TOKEN)]*(length - len(tokens)))
        else:
            tokens = tokens[:length]
        return tokens

    for instance in dataset:
        instance.fields['premise'].tokens = aux(
            instance.fields['premise'].tokens)
        instance.fields['hypothesis'].tokens = aux(
            instance.fields['hypothesis'].tokens)


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

    tokenizer = WordTokenizer()
    token_indexer = SingleIdTokenIndexer(lowercase_tokens=True)
    dataset_reader = SnliReader(
        tokenizer=tokenizer, token_indexers={'tokens': token_indexer})

    train_labeled_dataset_path = train_params.pop(
        'train_labeled_dataset_path')
    train_unlabeled_dataset_path = train_params.pop(
        'train_unlabeled_dataset_path', None)
    train_labeled_dataset = dataset_reader.read(train_labeled_dataset_path)
    truncate_or_pad_dataset(dataset=train_labeled_dataset, length=29)
    if train_unlabeled_dataset_path is not None:
        train_unlabeled_dataset = dataset_reader.read(
            train_unlabeled_dataset_path)
        truncate_or_pad_dataset(
            dataset=train_unlabeled_dataset, length=29)
    else:
        train_unlabeled_dataset = []

    valid_dataset = dataset_reader.read(
        train_params.pop('valid_dataset_path'))
    truncate_or_pad_dataset(valid_dataset, length=29)

    vocab = Vocabulary.from_instances(
        instances=train_labeled_dataset + train_unlabeled_dataset,
        max_vocab_size=train_params.pop_int('max_vocab_size', None))
    vocab.save_to_files(save_dir / 'vocab')

    labeled_batch_size = train_params.pop_int('labeled_batch_size')
    unlabeled_batch_size = train_params.pop_int('unlabeled_batch_size')
    labeled_iterator = BasicIterator(batch_size=labeled_batch_size)
    unlabeled_iterator = BasicIterator(batch_size=unlabeled_batch_size)
    labeled_iterator.index_with(vocab)
    unlabeled_iterator.index_with(vocab)

    if not train_unlabeled_dataset:
        unlabeled_iterator = None

    model = DeconvSNLIModel(params=model_params, vocab=vocab)
    optimizer = optim.Adam(params=model.parameters())
    summary_writer = SummaryWriter(log_dir=save_dir / 'log')

    trainer = SeparatedLVMTrainer(
        model=model,
        optimizer=optimizer,
        labeled_iterator=labeled_iterator,
        unlabeled_iterator=unlabeled_iterator,
        train_labeled_dataset=train_labeled_dataset,
        train_unlabeled_dataset=train_unlabeled_dataset,
        validation_dataset=valid_dataset,
        summary_writer=summary_writer,
        serialization_dir=save_dir,
        num_epochs=train_params.pop('num_epochs', 50),
        iters_per_epoch=len(train_labeled_dataset) // labeled_batch_size,
        write_summary_every=100,
        validate_every=2000,
        patience=2,
        clip_grad_max_norm=5,
        cuda_device=train_params.pop_int('cuda_device', 0)
    )
    trainer.train()


if __name__ == '__main__':
    main()
