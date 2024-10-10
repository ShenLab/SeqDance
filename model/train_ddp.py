# torchrun --nnodes=1 --nproc_per_node=4 train_ddp.py

import os
import time
import math
from itertools import cycle
import pandas as pd
import random
random.seed(0)  # Set random seed for reproducibility
import numpy as np
np.random.seed(0)  # Set numpy random seed for reproducibility
import h5py  # Handle the large feature dataset in HDF5 format
import torch
torch.manual_seed(0)  # Set PyTorch random seed for reproducibility
import torch.distributed as dist  # For distributed training
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP  # Wrapping model for distributed training
from transformers import AutoTokenizer  # SeqDance uses ESM2 tokenizer
from torch.optim import AdamW  # Optimizer
from torch.cuda.amp import autocast, GradScaler  # For Automatic Mixed Precision (AMP)
from torch.utils.data import Dataset, DataLoader  # Data handling
from torch.utils.data.distributed import DistributedSampler  # For distributed data loading

from config import config  # Configuration settings for the training process
from model import ESMwrap  # Wrap ESM for training
from utils import WarmupDecaySchedule, randomize_model, calculate_loss  # Utility functions
from dataset import ProteinDataset, collate_batch  # Dataset loading and batching

#############################################################
# DDP model train
#############################################################
def DDP_model(seqdance):
    # Initialize the distributed process group using NCCL (optimized for GPUs)
    dist.init_process_group("nccl")
    rank = dist.get_rank()  # Get the rank of the current process (its ID in the distributed setup)
    print(f"Start running DDP on rank {rank}.")

    # Determine the GPU to use based on the rank of the process
    device_id = rank % torch.cuda.device_count()

    # Move the model to the specified GPU
    seqdance = seqdance.to(device_id)

    # Wrap the model for distributed training, with gradients synchronized across devices
    ddp_model = DDP(seqdance, device_ids=[device_id], find_unused_parameters=True)

    # Print total number of trainable parameters on rank 1
    if rank == 1:
        print("total trainable params: ", sum(p.numel() for p in ddp_model.parameters() if p.requires_grad))

    # Create separate datasets for each source type and corresponding DataLoaders for training
    four_dataset = [
        ProteinDataset(df[df['source'].isin(source)].reset_index(drop=True), h5py_read, tokenizer, max_len) 
        for source in [['ATLAS', 'GPCRmd', 'PED'], ['IDRome'], ['proteinflow_pdb'], ['proteinflow_sabdab']]
    ]

    four_loader = [
        DataLoader(a_dataset, batch_size=batch_size, sampler=DistributedSampler(a_dataset), collate_fn=collate_batch, drop_last=True, num_workers=2, pin_memory=True) 
        for a_dataset in four_dataset
    ]

    # Initialize the AdamW optimizer and the learning rate scheduler
    optimizer = AdamW(ddp_model.parameters(), lr=1.0, betas=config['optimizer']['betas'], eps=config['optimizer']['epsilon'], weight_decay=config['optimizer']['weight_decay'])
    scheduler = WarmupDecaySchedule(optimizer, warmup_steps=config['optimizer']['warmup_step'], peak_lr=config['optimizer']['peak_lr'], total_steps=total_update)

    # Enable automatic mixed precision for faster training with lower memory usage
    scaler = GradScaler()

    # Initialize variables for tracking time, loss, and updates
    start_time = time.time()
    loss = 0
    samples = []
    loss_value = []
    n_update = 0
    batch_idx = 0

    # Start training loop, until the number of updates reaches the specified total
    while n_update <= total_update:
        # Every `save_per_update` steps, save model checkpoint and reinitialize data iterators
        if n_update % save_per_update == 0:
            # Set the epoch for distributed samplers, ensuring a new shuffle each epoch
            for a_loader in four_loader:
                a_loader.sampler.set_epoch(n_update)

            # Create iterators for cyclic loading of datasets
            four_cycle_iter = {
                'atlas_gpcrmd_ped': iter(cycle(four_loader[0])), 
                'idr': iter(cycle(four_loader[1])),
                'pdb': iter(cycle(four_loader[2])),
                'sabdab': iter(cycle(four_loader[3]))
            }
            # Save model state if on rank 0
            if rank == 0:        
                torch.save(ddp_model.module.state_dict(), f'update_{n_update}.tar')

        # Increment the batch index
        batch_idx += 1

        # Loop through the sources in the config, and fetch data from each source in a cyclic manner
        for source in config['training']['source_loop']:
            data = next(four_cycle_iter[source])

            # Move data to the correct GPU
            inputs = {"input_ids": data['input_ids'].to(device_id, non_blocking=True), "attention_mask": data['attention_mask'].to(device_id, non_blocking=True)}
            res_feat, pair_feat = data['res_feat'].to(device_id, non_blocking=True), data['pair_feat'].to(device_id, non_blocking=True)

            # Perform forward pass using mixed precision (AMP)
            with autocast():
                output = ddp_model(inputs)
                losses = calculate_loss(source, output, res_feat, pair_feat)

                # Calculate weighted loss for this batch
                loss_batch = (losses['res_loss'] + losses['pair_loss']) * loss_weight[source] / update_batch

            # Scale the loss, perform backpropagation
            scaler.scale(loss_batch).backward()

            # Accumulate loss for logging
            loss += loss_batch.item()
            loss_value.append([round(losses['res_loss'].item(), 6), round(losses['pair_loss'].item(), 6)])

        # Once enough batches are processed, update the model
        if batch_idx % update_batch == 0:
            # Unscale the gradients before performing gradient clipping
            scaler.unscale_(optimizer)

            # Clip gradients to prevent exploding gradients
            nn.utils.clip_grad_value_(ddp_model.parameters(), 0.5)

            # Step the optimizer and the learning rate scheduler
            scheduler.step()
            scaler.step(optimizer)
            scaler.update()

            # Zero the gradients for the next update
            optimizer.zero_grad()

            # Increment the update counter
            n_update += 1

            # Calculate time taken for the batch
            update_time = time.time() - start_time
            start_time = time.time()

            # Log the progress including time, loss, learning rate, and batch details
            print(f"rank: {rank}; n_update: {n_update}; Batch time: {update_time}; Loss: {loss}; Learning Rate: {scheduler.get_last_lr()[0]}; Loss_summary: {loss_value}")

            # Reset loss and value trackers for the next iteration
            loss = 0
            loss_value = []

    # Clean up the distributed process group after training completes
    dist.destroy_process_group()

if __name__ == "__main__":
#############################################################
# define the model and params
#############################################################
    esm2_select = 'model_35M'  # Select the ESM2 (35M)
    
    # use tokenizer of ESM2 model
    tokenizer = AutoTokenizer.from_pretrained(config[esm2_select]['model_id'])

    # Initialize the weights of the model (ESMwrap), donot use evolution information
    seqdance = ESMwrap(esm2_select)

    # Randomize model weights (useful for resetting any pre-trained weights)
    seqdance = randomize_model(seqdance)

    # Set the model to training mode (important for dropout layers, etc.)
    seqdance.train()

    # Load configuration parameters for training
    total_update = config['training']['total_update']  # Total number of updates
    save_per_update = config['training']['save_per_update']  # How often to save the model (in updates)
    loss_weight = config['training']['loss_weight']  # Loss weights for different datasets
    max_len = config['training']['max_len']  # Maximum sequence length for input data
    batch_size = config[esm2_select]['batch_size']  # Batch size for data loading
    update_batch = config[esm2_select]['update_batch']  # Number of batches per update (gradient step)
    
#############################################################
# load the dataset
#############################################################
    # Load training data
    df = pd.read_csv(config['file_path']['train_df_path'])
    h5py_read = h5py.File(config['file_path']['h5py_path'], 'r')

    # Load the evaluation dataset (and ensure it's excluded from the training set)
    eval_list = pd.read_csv(config['file_path']['eval_df_path'])['name'].tolist()
    df = df[~df['name'].isin(eval_list)].reset_index(drop=True)

    # Change directory to the specified save path (where models will be stored)
    os.chdir(config['file_path']['save_path'])

#############################################################
# distributed train
#############################################################
    # Start distributed training with DDP_model function
    DDP_model(seqdance)
