import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from itertools import cycle

# Import project-specific configurations and dataset-related classes/functions
from config import config  
from dataset import ProteinDataset, collate_batch  

#############################################################
# Learning Rate Warmup and Linear Decay Schedule
#############################################################
class WarmupDecaySchedule(LambdaLR):
    """
    Implements a learning rate schedule with:
    1. A warmup phase: The learning rate linearly increases to peak_lr over warmup_steps.
    2. A decay phase: The learning rate linearly decreases to 10% of peak_lr over decay_steps.
    3. A final phase: The learning rate remains constant at 10% of peak_lr.
    """
    def __init__(self, optimizer: Optimizer, model_select):
        self.warmup_steps = config['optimizer']['warmup_step']  # Number of warmup steps
        self.peak_lr = config['optimizer']['peak_lr']  # Maximum learning rate after warmup
        self.total_steps = config[model_select]['total_update']  # Total training steps
        self.decay_steps = int(self.total_steps * config['optimizer']['decay_step_percent'])  # Steps for linear decay

        super(WarmupDecaySchedule, self).__init__(optimizer, self.lr_lambda)

    def lr_lambda(self, step: int) -> float:
        """
        Computes the learning rate scaling factor based on the current step.
        """
        if step < self.warmup_steps:
            decay_factor = float(step) / float(self.warmup_steps)  # Linear increase
        elif step < self.decay_steps:
            decay_factor = 1 - 0.9 * (float(step - self.warmup_steps) / float(self.decay_steps - self.warmup_steps))  # Linear decay
        else:
            decay_factor = 0.1  # Fixed lower bound of 10% peak_lr
        
        return decay_factor * self.peak_lr  # Compute final learning rate

#############################################################
# Loss Function Definition
#############################################################
mseloss = nn.MSELoss()  # Mean Squared Error loss function

# Load loss weight from a CSV file, which contains the weight of each feature from different data sources
loss_weight = pd.read_csv(config['file_path']['loss_weight'], index_col=0)

def calculate_loss(source, output, res_feat, pair_feat):
    """
    Computes the loss for residue-level and pairwise dynamic features.
    
    Args:
        source (str): Data source name.
        output (dict): Model output containing predictions for various features.
        res_feat (torch.Tensor): Ground truth residue-level features.
        pair_feat (torch.Tensor): Ground truth pairwise features.
    
    Returns:
        dict: A dictionary with individual feature losses and total loss.
    """
    losses = {}  # Dictionary to store losses for different features

    # Create masks to filter out padding regions in res_feat and pair_feat
    valid_res_mask = (res_feat[:, :, 0] != -1)
    valid_pair_mask = (pair_feat[:, :, :, 0] != -1)

    if source in ['ATLAS_GPCRmd_PED_mdCATH', 'IDRome']:
        # Compute loss for residue-level features
        for feature in config['training']['res_feature_idx']:
            if 'nma' not in feature:  # Exclude normal mode analysis (NMA) features
                losses[feature] = torch.stack([
                    torch.sqrt(mseloss(output[feature][p][valid_res_mask[p]], 
                                       res_feat[p, :, config['training']['res_feature_idx'][feature]][valid_res_mask[p]]))
                    for p in range(len(res_feat))
                ]).mean() # the loss is the mean for all proteins in the batch

        # Compute loss for pairwise features
        for feature in config['training']['pair_feature_idx']:
            if 'nma' not in feature:
                losses[feature] = torch.stack([
                    torch.sqrt(mseloss(output[feature][p][valid_pair_mask[p]], 
                                       pair_feat[p, :, :, config['training']['pair_feature_idx'][feature]][valid_pair_mask[p]]))
                    for p in range(len(pair_feat))
                ]).mean()

    elif source in ['Proteinflow']:
        # Compute loss for three normal mode analysis (NMA) frequencies
        for k in range(3):  
            losses[f'nma_res{k+1}'] = torch.stack([
                torch.sqrt(mseloss(output[f'nma_res{k+1}'][p][valid_res_mask[p]], 
                                   res_feat[p, :, k][valid_res_mask[p]]))
                for p in range(len(res_feat))
            ]).mean()
            
            losses[f'nma_pair{k+1}'] = torch.stack([
                torch.sqrt(mseloss(output[f'nma_pair{k+1}'][p][valid_pair_mask[p]], 
                                   pair_feat[p, :, :, k][valid_pair_mask[p]]))
                for p in range(len(pair_feat))
            ]).mean()

    # Compute weighted sum of losses based on predefined weights
    loss = [losses[feature] * loss_weight.loc[feature, source] for feature in losses]
    losses['loss'] = torch.stack(loss).sum()

    # Rename loss keys to include data source prefix
    losses_rename = {f'{source}_{key}': losses[key] for key in losses}
    
    return losses_rename  # Return loss dictionary

#############################################################
# Get Dataloader Cycle Iterator for Three Data Sources
#############################################################
def get_dataloader_cycle_iter(df, h5py_read, esm2_select, max_len, batch_size, n_update, device_id):
    """
    Creates cyclic iterators for data loaders from three different protein data sources.

    Args:
        df (pd.DataFrame): Dataframe containing protein metadata.
        h5py_read (function): Function to read HDF5 data.
        esm2_select (str): Selection of ESM2 model.
        max_len (int): Maximum sequence length.
        batch_size (int): Batch size for training.
        n_update (int): Number of updates (used for setting seed).
        device_id (int): Device identifier (used for setting seed).

    Returns:
        dict: A dictionary mapping data sources to cyclic data loader iterators.
    """
    # Set random seed for reproducibility
    torch.manual_seed(n_update + device_id)
    torch.cuda.manual_seed(n_update + device_id)

    # Define dataset objects for three different protein data sources
    three_dataset = [
        ProteinDataset(df[df['source'].isin(source)].reset_index(drop=True), h5py_read, esm2_select, max_len)
        for source in [['ATLAS', 'GPCRmd', 'PED'], ['IDRome'], ['proteinflow_pdb', 'proteinflow_sabdab']]
    ]

    # Create DataLoader objects with DistributedSampler for parallel training
    three_loader = [
        DataLoader(a_dataset, batch_size=batch_size, sampler=DistributedSampler(a_dataset), 
                   collate_fn=collate_batch, drop_last=True, num_workers=2, pin_memory=True) 
        for a_dataset in three_dataset
    ]

    # Create cyclic iterators for each data source to allow infinite sampling
    three_cycle_iter = {
        'ATLAS_GPCRmd_PED_mdCATH': iter(cycle(three_loader[0])),
        'IDRome': iter(cycle(three_loader[1])),
        'Proteinflow': iter(cycle(three_loader[2]))
    }

    return three_cycle_iter  # Return dictionary of cyclic iterators
