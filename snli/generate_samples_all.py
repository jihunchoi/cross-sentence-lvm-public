import argparse
import json
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
from tqdm import tqdm

from snli.model import SNLIModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint', required=True)
    parser.add_argument('--lp-alpha', default=0.7, type=float)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--out', required=True)
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

    model_params['token_embedder']['pretrained_file'] = None
    model = SNLIModel(params=model_params, vocab=vocab)
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'),
                          strict=False)
    model.to(args.device)
    model.eval()

    iterator = BasicIterator(batch_size=args.batch_size)
    iterator.index_with(vocab)
    generator = iterator(valid_dataset, num_epochs=1, shuffle=False)
    label_index_to_token = vocab.get_index_to_token_vocabulary('labels')

    out_file = open(args.out, 'w')

    for batch in tqdm(generator):
        premise_tokens = batch['premise']['tokens']
        enc_embs = model.embed(premise_tokens.to(args.device))
        enc_mask = get_text_field_mask(batch['premise']).to(args.device)
        enc_hidden = model.encode(inputs=enc_embs, mask=enc_mask,
                                  drop_start_token=True)
        code, kld = model.sample_code_and_compute_kld(enc_hidden)
        pre_text = model.convert_to_readable_text(premise_tokens[:, 1:])
        label_tensor = batch['label'].to(args.device)
        generated = model.generate(
            code=code, label=label_tensor, max_length=25,
            beam_size=10, lp_alpha=args.lp_alpha)
        text = model.convert_to_readable_text(generated[:, 0])
        for pre_text_b, text_b, label_index_b in zip(pre_text, text, label_tensor):
            obj = {'sentence1': ' '.join(pre_text_b), 'sentence2': ' '.join(text_b),
                   'gold_label': label_index_to_token[label_index_b.item()]}
            out_file.write(json.dumps(obj))
            out_file.write('\n')


if __name__ == '__main__':
    main()
