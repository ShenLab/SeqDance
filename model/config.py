config = {
    "file_path": {
        "save_path": '/home/ch3849/ProDance/',
        "train_df_path":"SeqDance/dataset/all_dataset_ATLAS_GPCRmd_PED_IDRome_proteinflow.csv",
        "eval_df_path":"SeqDance/dataset/heldout_dataset_ATLAS_GPCRmd_PED_IDRome_proteinflow.csv",
        "h5py_path":"/home/ch3849/ProDance/data/train_data_all/feature_all_ATLAS_GPCRmd_PED_IDRome_proteinflow_newres.h5",
    },
    
    "training": {
        "dropout": 0.1,
        "max_len" : 512,
        "n_gpu": 6,
        "n_epoch": 3000,
        "total_update": 2e5,
        "save_per_update": 500,
        "source_loop": ['atlas_gpcrmd_ped'] + ['idr']*4 + ['pdb']*4 + ['sabdab'],
        "loss_weight": {'atlas_gpcrmd_ped': 1, 'idr': 0.7, 'pdb': 8, 'sabdab': 11}
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
        "model_id": "facebook/esm2_t12_35M_UR50D",
        "pair_in_feature": 240,
        "pair_out_feature": 13,
        "res_in_feature": 480,
        "res_out_feature": 50,
        "batch_size": 12,
        "update_batch": 1,
    },
}