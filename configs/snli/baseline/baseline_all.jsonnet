local NUM_LABELED = "120k";

{
    "train": {
        "train_labeled_dataset_path": "/hdd/datasets/snli/snli_1.0/snli_1.0_train.jsonl",
        "train_unlabeled_dataset_path": null,
        "valid_dataset_path": "/hdd/datasets/snli/snli_1.0/snli_1.0_dev.jsonl",
        "test_dataset_path": "/hdd/datasets/snli/snli_1.0/snli_1.0_test.jsonl",

        "batch_size": 32,

        "device": "cuda:0",
        "max_vocab_size": 20000,

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
