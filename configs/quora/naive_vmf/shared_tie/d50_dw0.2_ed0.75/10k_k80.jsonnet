
local NUM_LABELED = "10k";
local CODE_DIM = 50;
local VMF_KAPPA = 80;
local DISC_WEIGHT = 0.2;
local EMB_DROP = 0.75;
local DISC_DROP = 0.1;
local L2_WEIGHT = 0.002;

{
    "train": {
        "train_labeled_dataset_path": "data/quora/double/" + NUM_LABELED + "/labeled.txt",
        "train_unlabeled_dataset_path": "data/quora/double/" + NUM_LABELED + "/unlabeled.txt",
        "valid_dataset_path": "/hdd/datasets/quora/Quora_question_pair_partition/dev.tsv",
        "test_dataset_path": "/hdd/datasets/quora/Quora_question_pair_partition/test.tsv",

        "labeled_batch_size": 32,
        "unlabeled_batch_size": 32,

        "device": "cuda:0",
        "max_vocab_size": 10000,
        "num_epochs": 50
    },
    "model": {
        "token_embedder": {
            "embedding_dim": 300
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
