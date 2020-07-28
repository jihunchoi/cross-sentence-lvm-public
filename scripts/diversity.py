"""Measures diversity given a text file.

distinct-1 computes the ratio of unique unigrams over the
total number of words.
distinct-2 computes the ratio of unique bigrams.

Reference: A Diversity-Promoting Objective Function for Neural Conversation Models (Li et al. NAACL 2016)
"""

import argparse
import json
from collections import Counter

import nltk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', default=1, type=int, choices=[1, 2])
    parser.add_argument('--data', required=True)
    args = parser.parse_args()

    ngram_counter = {'contradiction': Counter(),
                     'neutral': Counter(),
                     'entailment': Counter(),
                     'all': Counter()}
    with open(args.data, 'r') as f:
        for line in f:
            obj = json.loads(line)
            hyp = obj['sentence2']
            words = hyp.split()
            ngrams = list(nltk.ngrams(words, n=args.type))
            if obj['gold_label'] not in ['entailment', 'contradiction', 'neutral']:
                continue
            ngram_counter['all'].update(ngrams)
            ngram_counter[obj['gold_label']].update(ngrams)

    print('---- distinct-{args.type} ----')
    for label, counter in ngram_counter.items():
        num_total_ngrams = sum(counter.values())
        num_unique_ngrams = len(counter.keys())
        print('------')
        print(label)
        print(f'# total ngrams: {num_total_ngrams}')
        print(f'# unique ngrams: {num_unique_ngrams}')
        print(f'distinct-{args.type} = {num_unique_ngrams / num_total_ngrams}')


if __name__ == '__main__':
    main()
