local NUM_LABELED = "28k";
local CODE_DIM = 50;
local VMF_KAPPA = 100;
local DISC_WEIGHT = 0.2;
local EMB_DROP = 0.25;
local DISC_DROP = 0.1;
local L2_WEIGHT = 0.002;
local CON_Y_WEIGHT = 0.0;
local CON_Z_WEIGHT = 1.0;

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
        "num_epochs": 20,

        "gumbel_anneal_rate": 1e-4,
        "lr": 1e-4,
        "aux_lr": 1e-4,

        "con_y_weight": CON_Y_WEIGHT,
        "con_z_weight": CON_Z_WEIGHT,
        "exclude_generator": EXCLUDE_GENERATOR,

        "pretrained_checkpoint_path": "trained/snli/naive_vmf/shared_tie/d50_dw0.2/vmf_28k_k100_ed0.25_dd0.1_wd0.002/model_state_best.th"
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
