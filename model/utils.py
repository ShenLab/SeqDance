import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from itertools import cycle

from config import config
from dataset import ProteinDataset, collate_batch

#############################################################
# learning rate Warmup linear Decay Schedule
#############################################################
class WarmupDecaySchedule(LambdaLR):
    def __init__(self, optimizer: Optimizer, model_select):
        self.warmup_steps = config['optimizer']['warmup_step']
        self.peak_lr = config['optimizer']['peak_lr']
        self.total_steps = config[model_select]['total_update']
        self.decay_steps = int(self.total_steps*config['optimizer']['decay_step_percent'])
        super(WarmupDecaySchedule, self).__init__(optimizer, self.lr_lambda)

    def lr_lambda(self, step: int) -> float:
        if step < self.warmup_steps:
            decay_factor = float(step) / float(self.warmup_steps)
        elif step < self.decay_steps:
            decay_factor = 1 - 0.9*(float(step - self.warmup_steps) / float(self.decay_steps - self.warmup_steps)) # linear decay
        else:
            decay_factor = 0.1
        return decay_factor * self.peak_lr
    
#############################################################
# loss
#############################################################
mseloss = nn.MSELoss()
# the weight of each feature from three data sources in the final loss
loss_weight = pd.read_csv(config['file_path']['loss_weight'], index_col=0)

def calculate_loss(source, output, res_feat, pair_feat):
    # create a dict to store the loss of each feature
    losses = {}
    # create the mask for paddings in res_feat and pair_feat
    valid_res_mask = (res_feat[:,:,0] != -1)
    valid_pair_mask = (pair_feat[:, :, :, 0] != -1)
    if source in ['ATLAS_GPCRmd_PED_mdCATH', 'IDRome']:
        for feature in config['training']['res_feature_idx']:
            if 'nma' not in feature:
                losses[feature] = torch.stack([torch.sqrt(mseloss(output[feature][p][valid_res_mask[p]], res_feat[p,:, config['training']['res_feature_idx'][feature]][valid_res_mask[p]])) for p in range(len(res_feat))]).mean()

        for feature in config['training']['pair_feature_idx']:
            if 'nma' not in feature:
                losses[feature] = torch.stack([torch.sqrt(mseloss(output[feature][p][valid_pair_mask[p]], pair_feat[p, :, :, config['training']['pair_feature_idx'][feature]][valid_pair_mask[p]])) for p in range(len(pair_feat))]).mean()

    elif source in ['Proteinflow']:
        for k in range(3): # three NMA frequencies
            losses[f'nma_res{k+1}'] = torch.stack([torch.sqrt(mseloss(output[f'nma_res{k+1}'][p][valid_res_mask[p]], res_feat[p,:, k][valid_res_mask[p]])) for p in range(len(res_feat))]).mean()
            losses[f'nma_pair{k+1}'] = torch.stack([torch.sqrt(mseloss(output[f'nma_pair{k+1}'][p][valid_pair_mask[p]], pair_feat[p, :, :, k][valid_pair_mask[p]])) for p in range(len(pair_feat))]).mean()
    
    loss = [losses[feature] * loss_weight.loc[feature, source] for feature in losses]
    losses['loss'] = torch.stack(loss).sum()
    losses_rename = {f'{source}_{key}': losses[key] for key in losses}
    return losses_rename

#############################################################
# get dataloader cycle iter for three sources
#############################################################
def get_dataloader_cycle_iter(df, h5py_read, esm2_select, max_len, batch_size, n_update, device_id):
    torch.manual_seed(n_update+device_id)
    torch.cuda.manual_seed(n_update+device_id)
    # define the dataset and dataloader
    three_dataset = [ProteinDataset(df[df['source'].isin(source)].reset_index(drop=True), h5py_read, esm2_select, max_len) 
                    for source in [['ATLAS', 'GPCRmd', 'PED'], ['IDRome'], ['proteinflow_pdb', 'proteinflow_sabdab']]]
    
    three_loader = [DataLoader(a_dataset, batch_size=batch_size, sampler=DistributedSampler(a_dataset), collate_fn=collate_batch, drop_last=True, num_workers=2, pin_memory=True) 
                   for a_dataset in three_dataset]
    
    three_cycle_iter = {'ATLAS_GPCRmd_PED_mdCATH': iter(cycle(three_loader[0])), 
                    'IDRome': iter(cycle(three_loader[1])),
                    'Proteinflow': iter(cycle(three_loader[2]))}
    
    return three_cycle_iter
