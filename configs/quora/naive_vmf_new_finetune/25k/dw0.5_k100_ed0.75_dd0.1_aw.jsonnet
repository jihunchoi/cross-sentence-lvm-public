
local NUM_LABELED = "25k";
local CODE_DIM = 50;
local VMF_KAPPA = 100;
local DISC_WEIGHT = 0.5;
local EMB_DROP = 0.75;
local DISC_DROP = 0.1;
local L2_WEIGHT = 0.002;
local CON_Y_WEIGHT = 1.0;
local CON_Z_WEIGHT = 1.0;
local CON_Z2_WEIGHT = 1.0;
local CON_AUTOWEIGHT = true;
local EXCLUDE_GENERATOR = false;

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
        "num_epochs": 100,

        "gumbel_anneal_rate": 1e-4,
        "lr": 1e-4,
        "aux_lr": 1e-4,

        "con_y_weight": CON_Y_WEIGHT,
        "con_z_weight": CON_Z_WEIGHT,
        "con_z2_weight": CON_Z2_WEIGHT,
        "con_autoweight": CON_AUTOWEIGHT,
        "exclude_generator": EXCLUDE_GENERATOR,

        "pretrained_checkpoint_path": "trained/quora/naive_vmf_new/25k/d50_dw0.5/k100_ed0.75_dd0.1/model_state_best.th"
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
