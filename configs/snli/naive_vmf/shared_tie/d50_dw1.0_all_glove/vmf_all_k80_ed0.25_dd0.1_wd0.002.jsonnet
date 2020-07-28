local NUM_LABELED = "all";
local CODE_DIM = 50;
local VMF_KAPPA = 80;
local DISC_WEIGHT = 1.0;
local EMB_DROP = 0.25;
local DISC_DROP = 0.1;
local L2_WEIGHT = 0.002;

{
    "train": {
        "train_labeled_dataset_path": "data/snli/" + NUM_LABELED + "/labeled.jsonl",
        "train_unlabeled_dataset_path": null,
        "valid_dataset_path": "/hdd/datasets/snli/snli_1.0/snli_1.0_dev.jsonl",
        "test_dataset_path": "/hdd/datasets/snli/snli_1.0/snli_1.0_test.jsonl",

        "labeled_batch_size": 32,
        "unlabeled_batch_size": 32,

        "device": "cuda:0",
        "num_epochs": 50
    },
    "model": {
        "token_embedder": {
            "embedding_dim": 300,
            "pretrained_file": "/hdd/pretrained/glove/glove.840B.300d.txt",
            "trainable": false
        },
        "code_dim": CODE_DIM,
        "code_dist_type": "vmf",
        "vmf_kappa": VMF_KAPPA,
        "discriminator_weight": DISC_WEIGHT,
        "shared_encoder": true,
        "tie_embedding": true,
        "emb_dropout": EMB_DROP,
        "disc_dropout": DISC_DROP,
        "l2_weight": L2_WEIGHT
    }
}
