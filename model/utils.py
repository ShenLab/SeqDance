import math
import random
random.seed(0)
import pickle
import h5py
import pandas as pd
import numpy as np
np.random.seed(0)

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.optimizer import Optimizer
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

from config import config

#############################################################
# random init esm2's pretrained weights
#############################################################
def randomize_model(model):
    for module_ in model.named_modules(): 
        if 'query' in module_[0] or 'key' in module_[0] or 'value' in module_[0]:
            module_[1].weight = nn.init.xavier_uniform_(module_[1].weight, gain=1 / math.sqrt(2))

        elif isinstance(module_[1],(torch.nn.Linear, torch.nn.Embedding)):
            module_[1].weight = nn.init.xavier_uniform_(module_[1].weight)

        elif isinstance(module_[1], nn.LayerNorm):
            module_[1].bias.data.zero_()
            module_[1].weight.data.fill_(1.0)

        if isinstance(module_[1], nn.Linear) and module_[1].bias is not None:
            module_[1].bias.data.zero_()
        
        if isinstance(module_[1], nn.Dropout):
            module_[1].p = config['training']['dropout']
            
    return model

#############################################################
# dataloader
#############################################################
def get_data_emb_label(df_source, source, h5py_read, tokenizer, batch_size, max_len, random_state):

    sub_df = df_source[source].sample(batch_size, replace=True, random_state=random_state)
    raw_input = tokenizer(list(sub_df['modify_seq']), return_tensors="pt", padding=True)

    batch_lens = raw_input['attention_mask'].sum(axis=1).tolist()
    batch_max_len = max(batch_lens)

    res_feat = [torch.tensor(h5py_read[f'{i}_res_feature'][:]) for i in sub_df['name']]
    pair_feat = [torch.tensor(h5py_read[f'{i}_pair_feature'][:]) for i in sub_df['name']]

    res_feat = pad_sequence(res_feat, batch_first=True, padding_value=-1)
    pair_feat = torch.stack([F.pad(sample, (0, 0, 0, batch_max_len-sample.size(0), 0, batch_max_len-sample.size(0)), mode='constant', value=-1) for sample in pair_feat])

    # random select a region of max_len if the protein is longer than max_len
    if batch_max_len > max_len:
        generate_random_nums = lambda x: 0 if x < max_len else random.randint(0, x - max_len)
        n_start = [generate_random_nums(abs(x)) for x in batch_lens]
        raw_input['input_ids'] = torch.stack([raw_input['input_ids'][i,n_start[i]:n_start[i]+max_len] for i in range(len(n_start))])
        raw_input['attention_mask'] = torch.stack([raw_input['attention_mask'][i,n_start[i]:n_start[i]+max_len] for i in range(len(n_start))])

        res_feat = torch.stack([res_feat[i,n_start[i]:n_start[i]+max_len,:] for i in range(len(n_start))])
        pair_feat = torch.stack([pair_feat[i, n_start[i]:n_start[i]+max_len, n_start[i]:n_start[i]+max_len, :] for i in range(len(n_start))])

    # use clamp to avoid extreme values in feature matrixes
    return {"sub_df":sub_df, "raw_input":raw_input, "res_feat": torch.clamp(res_feat, min=-2, max=2), "pair_feat": torch.clamp(pair_feat, min=-2, max=2)}

#############################################################
# learning rate Warmup linear Decay Schedule
#############################################################
class WarmupDecaySchedule(LambdaLR):
    def __init__(self, optimizer: Optimizer, warmup_steps: int, peak_lr: float, total_steps: int):
        self.warmup_steps = warmup_steps
        self.peak_lr = peak_lr
        self.total_steps = total_steps
        self.decay_steps = total_steps*config['optimizer']['decay_step_percent']
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
def calculate_loss(source, output, res_feat, pair_feat):
    if source in ['atlas_gpcrmd_ped', 'idr']:
        res_pred = output['res_pred']
        pair_pred = output['pair_pred']
        
        sasa_pred, sasa_feat = res_pred[:, :, :2], res_feat[:, :, :2]
        rmsf_pred, rmsf_feat = res_pred[:, :, 2], res_feat[:, :, 2]
        ss_pred, ss_feat = res_pred[:, :, 3:11], res_feat[:, :, 3:11]
        chi_pred, chi_feat = res_pred[:, :, 11:23], res_feat[:, :, 11:23]
        phi_pred, phi_feat = res_pred[:, :, 23:35], res_feat[:, :, 23:35]
        psi_pred, psi_feat = res_pred[:, :, 35:47], res_feat[:, :, 35:47]

        valid_mask = (rmsf_feat != -1)

        # Use torch.jit.fork to run loss calculations in parallel
        sasa_future = torch.jit.fork(mseloss, sasa_pred[valid_mask], sasa_feat[valid_mask])
        rmsf_future = torch.jit.fork(mseloss, rmsf_pred[valid_mask], rmsf_feat[valid_mask])
        ss_future = torch.jit.fork(mseloss, F.softmax(ss_pred[valid_mask], dim=-1), ss_feat[valid_mask])
        chi_future = torch.jit.fork(mseloss, F.softmax(chi_pred[valid_mask], dim=-1), chi_feat[valid_mask])
        phi_future = torch.jit.fork(mseloss, F.softmax(phi_pred[valid_mask], dim=-1), phi_feat[valid_mask])
        psi_future = torch.jit.fork(mseloss, F.softmax(psi_pred[valid_mask], dim=-1), psi_feat[valid_mask])


        valid_pair_mask = pair_feat[:, :, :, 0] != -1
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
        phi_loss = torch.jit.wait(psi_future)
        psi_loss = torch.jit.wait(psi_future)
        
        if source == 'atlas_gpcrmd_ped':
            res_loss = sasa_loss + rmsf_loss*8 + ss_loss*2 + chi_loss*2 + phi_loss*2 + psi_loss*2
        else:
            res_loss = sasa_loss*0.08 + rmsf_loss*0.4 + ss_loss*0.5 + chi_loss + phi_loss + psi_loss

        return {'res_loss':res_loss, 'pair_loss':pair_loss, 'md_res_loss':[sasa_loss, rmsf_loss, ss_loss, chi_loss, phi_loss, psi_loss]}

    else:
        res_pred = output['res_pred'][:, :, 47:50]
        pair_pred = output['pair_pred'][:, :, :, 10:13]

        # Compute valid masks
        valid_res_mask = res_feat != -1
        valid_pair_mask = pair_feat != -1

        # Fork the loss calculations to run them in parallel
        res_loss_future = torch.jit.fork(lambda pred, feat: mseloss(pred, feat) * 500, res_pred[valid_res_mask], res_feat[valid_res_mask])
        pair_loss_future = torch.jit.fork(mseloss, pair_pred[valid_pair_mask], pair_feat[valid_pair_mask])

        # Wait for the futures to complete and get the results
        res_loss = torch.jit.wait(res_loss_future)
        pair_loss = torch.jit.wait(pair_loss_future)

        return {'res_loss': res_loss, 'pair_loss': pair_loss}