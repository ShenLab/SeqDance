import math 
import random
random.seed(0)  # Set random seed for reproducibility

import h5py
import pandas as pd 
import numpy as np 
np.random.seed(0)  # Set seed for NumPy to ensure reproducibility

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR  # Learning rate scheduler
from torch.optim.optimizer import Optimizer
import torch.nn.functional as F

from config import config # Import configuration settings from the custom configuration file

#############################################################
# Function to randomize the weights of a pre-trained ESM2 model
#############################################################
def randomize_model(model):
    for module_ in model.named_modules():
        # Initialize the query, key, value weights with Xavier initialization scaled down by sqrt(2)
        if 'query' in module_[0] or 'key' in module_[0] or 'value' in module_[0]:
            module_[1].weight = nn.init.xavier_uniform_(module_[1].weight, gain=1 / math.sqrt(2))

        # Initialize Linear and Embedding layers with Xavier initialization
        elif isinstance(module_[1], (torch.nn.Linear, torch.nn.Embedding)):
            module_[1].weight = nn.init.xavier_uniform_(module_[1].weight)

        # Initialize LayerNorm layers: bias to zero and weights to 1.0
        elif isinstance(module_[1], nn.LayerNorm):
            module_[1].bias.data.zero_()
            module_[1].weight.data.fill_(1.0)

        # Zero the biases of Linear layers if they have a bias term
        if isinstance(module_[1], nn.Linear) and module_[1].bias is not None:
            module_[1].bias.data.zero_()

        # Set dropout probability to a value specified in the configuration file
        if isinstance(module_[1], nn.Dropout):
            module_[1].p = config['training']['dropout']

    return model


#############################################################
# Learning rate warmup and linear decay schedule
#############################################################
class WarmupDecaySchedule(LambdaLR):
    """
    A custom learning rate scheduler that implements a warmup phase followed by a linear decay.
    """
    def __init__(self, optimizer: Optimizer, warmup_steps: int, peak_lr: float, total_steps: int):
        self.warmup_steps = warmup_steps  # Number of warmup steps
        self.peak_lr = peak_lr  # Peak learning rate
        self.total_steps = total_steps  # Total number of training steps
        self.decay_steps = total_steps * config['optimizer']['decay_step_percent']  # Proportion of steps to apply linear decay
        super(WarmupDecaySchedule, self).__init__(optimizer, self.lr_lambda)

    def lr_lambda(self, step: int) -> float:
        """
        This function adjusts the learning rate depending on the current step.
        It increases the learning rate linearly during the warmup phase and decays it linearly thereafter.
        """
        if step < self.warmup_steps:
            decay_factor = float(step) / float(self.warmup_steps)  # Linear increase during warmup
        elif step < self.decay_steps:
            decay_factor = 1 - 0.9 * (float(step - self.warmup_steps) / float(self.decay_steps - self.warmup_steps))  # Linear decay
        else:
            decay_factor = 0.1  # Maintain a small constant learning rate at the end
        return decay_factor * self.peak_lr

#############################################################
# Loss function definitions
#############################################################
mseloss = nn.MSELoss()
def calculate_loss(source, output, res_feat, pair_feat):
    if source in ['atlas_gpcrmd_ped', 'idr']: # for MD data
        res_pred = output['res_pred']
        pair_pred = output['pair_pred']
        
        # Split predictions and features into individual components
        sasa_pred, sasa_feat = res_pred[:, :, :2], res_feat[:, :, :2]
        rmsf_pred, rmsf_feat = res_pred[:, :, 2], res_feat[:, :, 2]
        ss_pred, ss_feat = res_pred[:, :, 3:11], res_feat[:, :, 3:11]
        chi_pred, chi_feat = res_pred[:, :, 11:23], res_feat[:, :, 11:23]
        phi_pred, phi_feat = res_pred[:, :, 23:35], res_feat[:, :, 23:35]
        psi_pred, psi_feat = res_pred[:, :, 35:47], res_feat[:, :, 35:47]

        valid_mask = (rmsf_feat != -1) # we use -1 to indicate pad or linker regions, which should not be used in loss

        # Parallelize the calculation of MSE loss using torch.jit.fork for multiple features
        sasa_future = torch.jit.fork(mseloss, sasa_pred[valid_mask], sasa_feat[valid_mask])
        rmsf_future = torch.jit.fork(mseloss, rmsf_pred[valid_mask], rmsf_feat[valid_mask])
        ss_future = torch.jit.fork(mseloss, F.softmax(ss_pred[valid_mask], dim=-1), ss_feat[valid_mask])
        chi_future = torch.jit.fork(mseloss, F.softmax(chi_pred[valid_mask], dim=-1), chi_feat[valid_mask])
        phi_future = torch.jit.fork(mseloss, F.softmax(phi_pred[valid_mask], dim=-1), phi_feat[valid_mask])
        psi_future = torch.jit.fork(mseloss, F.softmax(psi_pred[valid_mask], dim=-1), psi_feat[valid_mask])


        valid_pair_mask = pair_feat[:, :, :, 0] != -1 # we use -1 to indicate pad or linker regions, which should not be used in loss
        pair_futures = []
        for i in range(10):
            pair_pred_i, pair_feat_i = pair_pred[:, :, :, i], pair_feat[:, :, :, i]

            if i < 9:
                if pair_feat_i.max() > 0:
                    nonzero_mask = pair_feat_i != 0
                    zero_mask = pair_feat_i == 0
                    pair_futures.append(torch.jit.fork(mseloss, pair_pred_i[valid_pair_mask & nonzero_mask], pair_feat_i[valid_pair_mask & nonzero_mask]))
                    pair_futures.append(torch.jit.fork(mseloss, pair_pred_i[valid_pair_mask & zero_mask], pair_feat_i[valid_pair_mask & zero_mask]))
                else:
                    pair_futures.append(torch.jit.fork(lambda pred, feat: 2 * mseloss(pred, feat), pair_pred_i[valid_pair_mask], pair_feat_i[valid_pair_mask]))
            else:
                weight = 6 if source == 'atlas_gpcrmd_ped' else 0.5
                pair_futures.append(torch.jit.fork(lambda pred, feat, w: w * mseloss(pred, feat), pair_pred_i[valid_pair_mask], pair_feat_i[valid_pair_mask], weight))

        # Wait for all pair futures to complete and get the results
        pair_losses = []
        for future in pair_futures:
            pair_losses.append(torch.jit.wait(future))

        pair_loss = torch.sum(torch.stack(pair_losses))

        # Wait for all futures to complete and get the results
        sasa_loss = torch.jit.wait(sasa_future)
        rmsf_loss = torch.jit.wait(rmsf_future)
        ss_loss = torch.jit.wait(ss_future)
        chi_loss = torch.jit.wait(chi_future)
        phi_loss = torch.jit.wait(phi_future)
        psi_loss = torch.jit.wait(psi_future)
        
        # we use different weights for different features in different sources, please refer to our manuscript for details
        if source == 'atlas_gpcrmd_ped':
            res_loss = sasa_loss + rmsf_loss*8 + ss_loss*2 + chi_loss*2 + phi_loss*2 + psi_loss*2
        else:
            res_loss = sasa_loss*0.08 + rmsf_loss*0.4 + ss_loss*0.5 + chi_loss + phi_loss + psi_loss

        return {'res_loss':res_loss, 'pair_loss':pair_loss, 'md_res_loss':[sasa_loss, rmsf_loss, ss_loss, chi_loss, phi_loss, psi_loss]}

    else: # for NMA loss
        res_pred = output['res_pred'][:, :, 47:50]
        pair_pred = output['pair_pred'][:, :, :, 10:13]

        # we use -1 to indicate pad or linker regions, which should not be used in loss
        valid_res_mask = res_feat != -1
        valid_pair_mask = pair_feat != -1

        # Fork the loss calculations to run them in parallel
        res_loss_future = torch.jit.fork(lambda pred, feat: mseloss(pred, feat) * 500, res_pred[valid_res_mask], res_feat[valid_res_mask])
        pair_loss_future = torch.jit.fork(mseloss, pair_pred[valid_pair_mask], pair_feat[valid_pair_mask])

        # Wait for the futures to complete and get the results
        res_loss = torch.jit.wait(res_loss_future)
        pair_loss = torch.jit.wait(pair_loss_future)

        return {'res_loss': res_loss, 'pair_loss': pair_loss}