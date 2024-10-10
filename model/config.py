config = {
    "file_path": {
        "save_path": '../dataset/',
        "train_df_path":"../dataset/all_dataset_ATLAS_GPCRmd_PED_IDRome_proteinflow.csv",
        "eval_df_path":"../dataset/heldout_dataset_ATLAS_GPCRmd_PED_IDRome_proteinflow.csv",
        "h5py_path":"if you want to use this file (~100GB), please contact us",
    },
    
    "training": {
        "dropout": 0.1,
        "max_len" : 512,
        "n_gpu": 6,
        "n_epoch": 3000,
        "total_update": 2e5,
        "save_per_update": 500,
        "source_loop": ['atlas_gpcrmd_ped'] + ['idr']*4 + ['pdb']*4 + ['sabdab'], # adjust for different number of samples, please refer to the paper for details.
        "loss_weight": {'atlas_gpcrmd_ped': 1, 'idr': 0.7, 'pdb': 8, 'sabdab': 11} # adjust for different number of samples, please refer to the paper for details.
    },

    "optimizer": {
        "peak_lr": 1e-4,
        "epsilon": 1e-8,
        "betas": (0.9,0.98),
        "weight_decay": 0.01,
        "warmup_step": 5000,
        "decay_step_percent": 0.9,
    },

    "model_35M": {
        "model_id": "facebook/esm2_t12_35M_UR50D", # use the architecture of ESM2 (35M)
        "pair_in_feature": 240, # number of attention maps
        "pair_out_feature": 13, # number of pairwise features
        "res_in_feature": 480, # dim of final residue embedding
        "res_out_feature": 50, # number of residue-level features
        "batch_size": 12, 
        "update_batch": 1,
    },
}