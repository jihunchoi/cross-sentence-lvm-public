import argparse
import random
from pathlib import Path
import torch

from allennlp.data.token_indexers.single_id_token_indexer import \
    SingleIdTokenIndexer

from allennlp.common.params import Params
from allennlp.data.dataset_readers import SnliReader
from allennlp.data.iterators.basic_iterator import BasicIterator
from allennlp.data.tokenizers.word_tokenizer import WordTokenizer
from allennlp.data.vocabulary import Vocabulary
from allennlp.nn.util import get_text_field_mask

from snli.model import SNLIModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint', required=True)
    parser.add_argument('--lp-alpha', default=0.7, type=float)
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

    tokenizer = WordTokenizer(start_tokens=['<s>'], end_tokens=['</s>'],)
    token_indexer = SingleIdTokenIndexer(lowercase_tokens=True)
    dataset_reader = SnliReader(
        tokenizer=tokenizer, token_indexers={'tokens': token_indexer})

    valid_dataset = dataset_reader.read(
        train_params.pop('valid_dataset_path'))
    vocab = Vocabulary.from_files(vocab_dir)
    random.shuffle(valid_dataset)

    model_params['token_embedder']['pretrained_file'] = None
    model = SNLIModel(params=model_params, vocab=vocab)
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'),
                          strict=False)
    model.eval()

    iterator = BasicIterator(batch_size=1)
    iterator.index_with(vocab)
    generator = iterator(valid_dataset)

    for i in range(10):
        batch = next(generator)
        label_token_to_index = vocab.get_token_to_index_vocabulary('labels')
        print('----')
        print(' '.join(model.convert_to_readable_text(batch['premise']['tokens'])[0]))
        for label, label_index in label_token_to_index.items():
            label_tensor = torch.tensor([label_index])
            enc_embs = model.embed(batch['premise']['tokens'])
            enc_mask = get_text_field_mask(batch['premise'])
            enc_hidden = model.encode(inputs=enc_embs, mask=enc_mask,
                                      drop_start_token=True)
            code, kld = model.sample_code_and_compute_kld(enc_hidden)
            generated = model.generate(
                code=code, label=label_tensor, max_length=enc_mask.sum(1) * 2,
                beam_size=10, lp_alpha=args.lp_alpha)
            text = model.convert_to_readable_text(generated[:, 0])[0]
            print(label)
            print(' '.join(text))


if __name__ == '__main__':
    main()
