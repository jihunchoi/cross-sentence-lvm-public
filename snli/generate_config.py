import argparse
from pathlib import Path

TEMPLATE = '''
local NUM_LABELED = "{num_labeled}";
local CODE_DIM = {code_dim};
local VMF_KAPPA = {vmf_kappa};
local DISC_WEIGHT = {disc_weight};
local EMB_DROP = {emb_drop};
local DISC_DROP = {disc_drop};
local L2_WEIGHT = 0.002;

{{
    "train": {{
        "train_labeled_dataset_path": "data/snli/" + NUM_LABELED + "/labeled.jsonl",
        "train_unlabeled_dataset_path": "data/snli/" + NUM_LABELED + "/unlabeled.jsonl",
        "valid_dataset_path": "/hdd/datasets/snli/snli_1.0/snli_1.0_dev.jsonl",
        "test_dataset_path": "/hdd/datasets/snli/snli_1.0/snli_1.0_test.jsonl",

        "labeled_batch_size": 32,
        "unlabeled_batch_size": 32,

        "device": "cuda:0",
        "max_vocab_size": 20000,
        "num_epochs": 50
    }},
    "model": {{
        "token_embedder": {{
            "embedding_dim": 300
        }},
        "code_dim": CODE_DIM,
        "code_dist_type": "vmf",
        "vmf_kappa": VMF_KAPPA,
        "discriminator_weight": DISC_WEIGHT,
        "shared_encoder": true,
        "tie_embedding": true,
        "emb_dropout": EMB_DROP,
        "disc_dropout": DISC_DROP,
        "l2_weight": L2_WEIGHT
    }}
}}
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-labeled', required=True)
    parser.add_argument('--code-dim', default=300, type=int)
    parser.add_argument('--vmf-kappa', required=True, type=int)
    parser.add_argument('--disc-weight', required=True, type=float)
    parser.add_argument('--emb-drop', default=0.5, type=float)
    parser.add_argument('--disc-drop', default=0.1, type=float)
    parser.add_argument('--save', required=True)
    parser.add_argument('--verbose', default=False, action='store_true')
    args = parser.parse_args()

    config = TEMPLATE.format(**args.__dict__)
    if args.verbose:
        print(config)

    save_path = Path(args.save)
    with open(save_path, 'w') as f:
        f.write(config)


if __name__ == '__main__':
    main()
