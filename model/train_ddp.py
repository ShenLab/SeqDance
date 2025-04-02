import os
import time
import pandas as pd
import random
import json
import shutil
import numpy as np
import h5py
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.amp import autocast, GradScaler

from config import config
from model import ESMwrap
from utils import WarmupDecaySchedule, calculate_loss, get_dataloader_cycle_iter

# Set random seeds for reproducibility
torch.manual_seed(config['training']['random_seed'])
torch.cuda.manual_seed_all(config['training']['random_seed'])
np.random.seed(config['training']['random_seed'])
random.seed(config['training']['random_seed'])

#############################################################
# Distributed Data Parallel (DDP) model training function
#############################################################
def DDP_model(esm2_select, model_select, dance_model, df, h5py_read, total_update, short_update, save_per_update, get_dataloader_per_update, save_dir):
    # Initialize distributed training environment using NCCL backend
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    print(f"Start running DDP on rank {rank}.")

    # Assign device based on rank
    device_id = rank % torch.cuda.device_count()
    dance_model = dance_model.to(device_id)
    ddp_model = DDP(dance_model, device_ids=[device_id], find_unused_parameters=True)

    # Rank 0 handles saving model and setup: mkdir, copy the code, save the initial model
    if rank == 0:
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            # Copy the code to the save directory
            shutil.copytree('../model/', os.path.join(save_dir, 'model'))
            # Create a directory to save checkpoints
            os.makedirs(os.path.join(save_dir, "checkpoints"), exist_ok=True)

        # save the initial model and print the total trainable parameters
        chk_path = os.path.join(save_dir, 'checkpoints', f'update_0.pt')
        torch.save(ddp_model.module.state_dict(), chk_path)
        print("total trainable params: ",sum(p.numel() for p in ddp_model.parameters() if p.requires_grad))

    # Define optimizer and learning rate scheduler
    optimizer = AdamW(ddp_model.parameters(), lr=1.0, betas=config['optimizer']['betas'], eps=config['optimizer']['epsilon'], weight_decay=config['optimizer']['weight_decay'])
    scheduler = WarmupDecaySchedule(optimizer, model_select)

    # Automatic Mixed Precision (AMP) scaler for training efficiency
    scaler = GradScaler('cuda')

    start_time = time.time()
    n_update = 0 # update counter
    n_epoch = 0
    n_batch_in_epoch = 0
    max_len = config[model_select]['max_len_short'] # start with short sequence length, change to long length after short_update

    while n_update <= total_update:
        # Refresh dataloader periodically and adjust max sequence length
        if (n_batch_in_epoch/config[model_select][f'update_batch_{max_len}']) % get_dataloader_per_update == 0:
            log = {}
            n_epoch += 1
            n_batch_in_epoch = 0
            # change the max_len
            if n_update >= short_update:
                max_len = config[model_select]['max_len_long']

            # get the dataloader of three sources
            three_cycle_iter = get_dataloader_cycle_iter(df, h5py_read, esm2_select, max_len, config[model_select][f'batch_size_{max_len}'], n_update, device_id)

        # TRAINING: one batch contains samples from three sources
        for source in three_cycle_iter:
            # get the data and move it to the device
            data = next(three_cycle_iter[source])
            inputs = {"input_ids": data['input_ids'].to(device_id, non_blocking=True), "attention_mask": data['attention_mask'].to(device_id, non_blocking=True)}
            res_feat, pair_feat = data['res_feat'].to(device_id, non_blocking=True), data['pair_feat'].to(device_id, non_blocking=True)
            
            # forward pass, calculate the loss, and backward pass
            with autocast(dtype=torch.float16, device_type='cuda'):
                output = ddp_model(inputs)
                losses = calculate_loss(source, output, res_feat, pair_feat)
                loss_batch = losses[f'{source}_loss'] / config[model_select][f'update_batch_{max_len}'] # devide by update_batch to get the mean loss
                    
            scaler.scale(loss_batch).backward()

            # mean loss per report (if report_per_update is 20, log the mean loss in the last 20 updates)
            for feature in losses:
                if feature in log.keys():
                    log[feature] += losses[feature].item()/config[model_select][f'update_batch_{max_len}']/config['training']['report_per_update']
                else:
                    log[feature] = losses[feature].item()/config[model_select][f'update_batch_{max_len}']/config['training']['report_per_update']
        
        # update the batch counter after the for loop of three sources
        n_batch_in_epoch += 1

        # update the model
        if n_batch_in_epoch % config[model_select][f'update_batch_{max_len}'] == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_value_(ddp_model.parameters(), 0.5)
            scheduler.step()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            n_update += 1
            
            # logging
            if n_update % config['training']['report_per_update'] == 0:
                update_time = time.time() - start_time
                start_time = time.time()

                log = {
                    "rank": rank,
                    "n_epoch": n_epoch,
                    "n_update": n_update,
                    "n_batch_in_epoch": n_batch_in_epoch,
                    "Batch time": update_time,
                    "Learning Rate": scheduler.get_last_lr()[0],
                    "Max_len": max_len,
                    **log
                }

                # write the log to file
                with open(f"{save_dir}/training_log.json", "a") as f:
                    json.dump(log, f)
                    f.write('\n')
                log = {}

            # save the model
            if n_update % save_per_update == 0:
                chk_path = os.path.join(save_dir, 'checkpoints', f'update_{n_update}.pt')
                if rank == 0 and not os.path.exists(chk_path):
                    torch.save(ddp_model.module.state_dict(), chk_path)

    dist.destroy_process_group()

if __name__ == "__main__":
#############################################################
# define the model
#############################################################
    esm2_select = 'model_35M'
    # model_select = 'seqdance' # seqdance: 35M, train all parameters, using attention to predict pair features
    model_select = 'esmdance' # esmdance: 35M, freeze the ESM parameters, using attention to predict pair features
    dance_model = ESMwrap(esm2_select, model_select)
    dance_model.train() # use this to activate dropout

#############################################################
# load the dataset and define key hyperparameters
#############################################################
    df = pd.read_csv(config['file_path']['train_df_path'])
    df = df[(df['label'] == 'train')]
    h5py_read = h5py.File(config['file_path']['h5py_path'], 'r')

    save_dir = config['file_path']['save_dir']
    total_update = config[model_select]['total_update']
    short_update = config[model_select]['short_update']
    save_per_update = config['training']['save_per_update']
    get_dataloader_per_update = config['training']['get_dataloader_per_update']

#############################################################
# training
#############################################################
    DDP_model(esm2_select, model_select, dance_model, df, h5py_read, total_update, short_update, save_per_update, get_dataloader_per_update, save_dir)