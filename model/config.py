config = {
    "file_path": {
        "save_dir": "save_dir",
        "train_df_path":"download this from https://huggingface.co/datasets/ChaoHou/protein_dynamic_properties",
        "h5py_path": "download this from https://huggingface.co/datasets/ChaoHou/protein_dynamic_properties",
        "loss_weight": "download this from https://huggingface.co/datasets/ChaoHou/protein_dynamic_properties",
    },
    
    "training": {
        "random_seed": 42,
        "dropout": 0.1,
        "n_gpu": 4,
        "save_per_update": 1000,
        "report_per_update": 20, # report the mean loss in the last 20 updates
        "get_dataloader_per_update": 200, # about 212 updates per epoch, use 200 for better save the model and logging
        "res_feature_idx": {'sasa_mean':0, 'sasa_std':1, 'rmsf_nor':2, 'ss':range(3,11), 'chi':range(11,23), 'phi':range(23,35), 'psi':range(35,47), 'nma_res1':47, 'nma_res2':48, 'nma_res3':49},
        "pair_feature_idx": {'vdw':0, 'hbbb':1, 'hbsb':2, 'hbss':3, 'hp':4, 'sb':5, 'pc':6, 'ps':7, 'ts':8, 'corr':9, 'nma_pair1':10, 'nma_pair2':11, 'nma_pair3':12},
    },

    "seqdance": {
        "freeze_esm": False,
        "randomize_esm": True,
        "max_len_short" : 256,
        "max_len_long" : 1024,
        "total_update": 200_000,
        "short_update": 160_000,
        "batch_size_256": 16,
        "update_batch_256": 2,
        "batch_size_1024": 2,
        "update_batch_1024": 16,
    },

    "esmdance": {
        "freeze_esm": True,
        "randomize_esm": False,
        "max_len_short" : 256,
        "max_len_long" : 1024,
        "total_update": 60_000,
        "short_update": 40_000,
        "batch_size_256": 16,
        "update_batch_256": 2,
        "batch_size_1024": 2,
        "update_batch_1024": 16,
    },

    "optimizer": {
        "peak_lr": 1e-4,
        "epsilon": 1e-8,
        "betas": (0.9,0.98),
        "weight_decay": 0.01,
        "warmup_step": 2000,
        "decay_step_percent": 0.9,
    },

    "model_35M": {
        "model_id": "facebook/esm2_t12_35M_UR50D",
        
        "atten_dim": 240,
        "embed_dim": 480,
        "pair_out_dim": 13,
        "res_out_dim": 50,
    },

}