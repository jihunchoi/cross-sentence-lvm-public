local NUM_LABELED = "59k";
local CODE_DIM = 50;
local VMF_KAPPA = 100;
local DISC_WEIGHT = 1.0;
local EMB_DROP = 0.75;
local DISC_DROP = 0.1;
local L2_WEIGHT = 0.002;
local SAMPLING = false;
local CODE_DIST = "vmf";

{
    "train": {
        "train_labeled_dataset_path": "data/snli/" + NUM_LABELED + "/labeled.jsonl",
        "train_unlabeled_dataset_path": "data/snli/" + NUM_LABELED + "/unlabeled.jsonl",
        "valid_dataset_path": "/hdd/datasets/snli/snli_1.0/snli_1.0_dev.jsonl",
        "test_dataset_path": "/hdd/datasets/snli/snli_1.0/snli_1.0_test.jsonl",

        "labeled_batch_size": 32,
        "unlabeled_batch_size": 32,

        "device": "cuda:0",
        "max_vocab_size": 20000,
        "num_epochs": 50,
        "kl_anneal_rate": 1e-6
    },
    "model": {
        "token_embedder": {
            "embedding_dim": 300
        },
        "code_dim": CODE_DIM,
        "code_dist_type": CODE_DIST,
        "vmf_kappa": VMF_KAPPA,
        "discriminator_weight": DISC_WEIGHT,
        "shared_encoder": false,
        "tie_embedding": true,
        "emb_dropout": EMB_DROP,
        "disc_dropout": DISC_DROP,
        "l2_weight": L2_WEIGHT,
        "use_sampling": SAMPLING
    }
}