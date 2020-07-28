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
from allennlp.data.instance import Instance
from allennlp.data.token_indexers.single_id_token_indexer import \
    SingleIdTokenIndexer
from allennlp.data.dataset_readers import SnliReader
from allennlp.data.iterators.basic_iterator import BasicIterator
from allennlp.data.tokenizers.word_tokenizer import WordTokenizer
from allennlp.data.vocabulary import Vocabulary
from tensorboardX import SummaryWriter

from snli.model import SNLIModel
from snli.trainers import FineTuningTrainer


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

    tokenizer = WordTokenizer(start_tokens=['<s>'], end_tokens=['</s>'])
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

    labeled_batch_size = train_params.pop_int('labeled_batch_size')
    unlabeled_batch_size = train_params.pop_int('unlabeled_batch_size')
    labeled_iterator = BasicIterator(batch_size=labeled_batch_size)
    unlabeled_iterator = BasicIterator(batch_size=unlabeled_batch_size)
    labeled_iterator.index_with(vocab)
    unlabeled_iterator.index_with(vocab)

    if not train_unlabeled_dataset:
        unlabeled_iterator = None

    pretrained_checkpoint_path = train_params.pop(
        'pretrained_checkpoint_path', None)
    model = SNLIModel(params=model_params, vocab=vocab)
    if pretrained_checkpoint_path:
        model.load_state_dict(
            torch.load(pretrained_checkpoint_path, map_location='cpu'))
    model.add_finetune_parameters(
        con_autoweight=train_params.pop_bool('con_autoweight', False),
        con_y_weight=train_params.pop_float('con_y_weight'),
        con_z_weight=train_params.pop_float('con_z_weight'),
        con_z2_weight=train_params.pop_float('con_z2_weight'))

    main_optimizer = optim.Adam(
        params=model.finetune_main_parameters(
            exclude_generator=train_params.pop_bool('exclude_generator')),
        lr=train_params.pop_float('lr', 1e-3)
    )
    aux_optimizer = optim.Adam(params=model.finetune_aux_parameters(),
                               lr=train_params.pop_float('aux_lr', 1e-4))

    summary_writer = SummaryWriter(log_dir=save_dir / 'log')

    kl_anneal_rate = train_params.pop_float('kl_anneal_rate', None)
    if kl_anneal_rate is None:
        kl_weight_scheduler = None
    else:
        kl_weight_scheduler = (
            lambda step: min(1.0, kl_anneal_rate * step)
        )
        model.kl_weight = 0.0

    gumbel_anneal_rate = train_params.pop_float('gumbel_anneal_rate', None)
    if gumbel_anneal_rate is None:
        gumbel_temperature_scheduler = None
    else:
        gumbel_temperature_scheduler = (
            lambda step: max(0.1, 1.0 - gumbel_anneal_rate*step)
        )
        model.gumbel_temperature = 1.0
    iters_per_epoch = train_params.pop_int(
        'iters_per_epoch', len(train_labeled_dataset) // labeled_batch_size)

    trainer = FineTuningTrainer(
        model=model,
        main_optimizer=main_optimizer,
        aux_optimizer=aux_optimizer,
        labeled_iterator=labeled_iterator,
        unlabeled_iterator=unlabeled_iterator,
        train_labeled_dataset=train_labeled_dataset,
        train_unlabeled_dataset=train_unlabeled_dataset,
        validation_dataset=valid_dataset,
        summary_writer=summary_writer,
        serialization_dir=save_dir,
        num_epochs=train_params.pop_int('num_epochs', 50),
        iters_per_epoch=iters_per_epoch,
        write_summary_every=100,
        validate_every=1000,
        patience=train_params.pop_int('patience', 5),
        clip_grad_max_norm=train_params.pop_float('grad_max_norm', 5.0),
        kl_weight_scheduler=kl_weight_scheduler,
        gumbel_temperature_scheduler=gumbel_temperature_scheduler,
        cuda_device=train_params.pop_int('cuda_device', 0),
    )
    trainer.train()


if __name__ == '__main__':
    main()
