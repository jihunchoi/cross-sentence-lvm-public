local NUM_LABELED = "1k";

{
    "train": {
        "train_labeled_dataset_path": "data/quora/single/" + NUM_LABELED + "/labeled.txt",
        "train_unlabeled_dataset_path": "data/quora/single/" + NUM_LABELED + "/unlabeled.txt",
        "valid_dataset_path": "/hdd/datasets/quora/Quora_question_pair_partition/dev.tsv",
        "test_dataset_path": "/hdd/datasets/quora/Quora_question_pair_partition/test.tsv",

        "batch_size": 32,

        "device": "cuda:0",
        "max_vocab_size": 10000,

        "num_epochs": 50
    },
    "model": {
        "token_embedder": {
            "embedding_dim": 300
        },
        "emb_dropout": 0.25,
        "disc_dropout": 0.1,
        "l2_weight": 0.002
    }
}
