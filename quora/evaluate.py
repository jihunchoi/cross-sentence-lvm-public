import argparse
import random
from pathlib import Path
from pprint import pprint

import torch

from allennlp.data.token_indexers import SingleIdTokenIndexer

from allennlp.common.params import Params
from allennlp.data.dataset_readers import QuoraParaphraseDatasetReader
from allennlp.data.iterators.basic_iterator import BasicIterator
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter
from allennlp.data.tokenizers.word_tokenizer import WordTokenizer
from allennlp.data.vocabulary import Vocabulary
from allennlp.nn.util import move_to_device
from tqdm import tqdm

from quora.model import QuoraModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint', required=True)
    parser.add_argument('--cuda-device', default=0, type=int)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    checkpoint_dir = checkpoint_path.parent
    params_path = checkpoint_dir / 'params.json'
    vocab_dir = checkpoint_dir / 'vocab'

    params = Params.from_file(params_path)
    train_params, model_params = params.pop('train'), params.pop('model')

    tokenizer = WordTokenizer(word_splitter=JustSpacesWordSplitter(),
                              start_tokens=['<s>'], end_tokens=['</s>'])
    token_indexer = SingleIdTokenIndexer(lowercase_tokens=True)
    dataset_reader = QuoraParaphraseDatasetReader(
        tokenizer=tokenizer, token_indexers={'tokens': token_indexer})

    valid_dataset = dataset_reader.read(
        train_params.pop('valid_dataset_path'))
    test_dataset = dataset_reader.read(
        train_params.pop('test_dataset_path'))
    vocab = Vocabulary.from_files(vocab_dir)
    random.shuffle(valid_dataset)

    model_params['token_embedder']['pretrained_file'] = None
    model = QuoraModel(params=model_params, vocab=vocab)
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'),
                          strict=False)
    model.to(args.cuda_device)
    model.eval()

    torch.set_grad_enabled(False)

    iterator = BasicIterator(batch_size=32)
    iterator.index_with(vocab)

    for dataset in (valid_dataset, test_dataset):
        generator = iterator(dataset, shuffle=False, num_epochs=1)
        model.get_metrics(reset=True)
        for batch in tqdm(generator):
            batch = move_to_device(batch, cuda_device=args.cuda_device)
            model(premise=batch['premise'],
                  hypothesis=batch['hypothesis'],
                  label=batch['label'])
        metrics = model.get_metrics()
        pprint(metrics)


if __name__ == '__main__':
    main()
