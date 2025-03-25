import os
import pandas as pd
import numpy as np
import h5py
import json
import torch
from config import config
from model import ESMwrap
from utils import calculate_loss
from dataset import ProteinDataset

# for evaluation get the mean loss in three dataset
def get_mean_loss_in_three_dataset(dance_model, three_dataset):
    mean_loss = {}
    for source in three_dataset:
        for data in three_dataset[source]:
            data = {k: v.unsqueeze(0) for k, v in data.items()}
            inputs = {"input_ids": data['input_ids'].to(device, non_blocking=True), "attention_mask": data['attention_mask'].to(device, non_blocking=True)}
            res_feat, pair_feat = data['res_feat'].to(device, non_blocking=True), data['pair_feat'].to(device, non_blocking=True)

            with torch.no_grad():
                output = dance_model(inputs)
                losses = calculate_loss(source, output, res_feat, pair_feat)

            for feature in losses:
                if feature in mean_loss.keys():
                     mean_loss[feature] += losses[feature].item()/len(three_dataset[source])
                else:
                     mean_loss[feature] = losses[feature].item()/len(three_dataset[source])
    
    return mean_loss

# load the model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

esm2_select = 'model_35M'
model_select = 'seqdance'
# model_select = 'esmdance'
dance_model = ESMwrap(esm2_select, model_select)

# load the dataset
df = pd.read_csv(config['file_path']['train_df_path'])
df = df[(df['label'] == 'test')]

h5py_read = h5py.File(config['file_path']['h5py_path'], 'r')

max_len = 1024

three_dataset = {
    'ATLAS_GPCRmd_PED_mdCATH': ProteinDataset(df[df['source'].isin(['ATLAS', 'GPCRmd', 'PED', 'mdCATH'])].reset_index(drop=True), h5py_read, esm2_select, max_len, start_from_N_terminal=True), 
    'IDRome': ProteinDataset(df[df['source'].isin(['IDRome'])].reset_index(drop=True), h5py_read, esm2_select, max_len, start_from_N_terminal=True),
    'Proteinflow': ProteinDataset(df[df['source'].isin(['proteinflow_pdb', 'proteinflow_sabdab'])].reset_index(drop=True), h5py_read, esm2_select, max_len, start_from_N_terminal=True)
}

# evaluation
model_date = '250217'

for chk in range(190_000, 200_000+1, 1000):
    checkpoint = torch.load(f'/nfs/user/Users/ch3849/ProDance/model/{model_date}/checkpoints/update_{chk}.pt')
    dance_model.load_state_dict(checkpoint, strict=False,)
    dance_model = dance_model.to(device)
    dance_model.eval()
    mean_loss = get_mean_loss_in_three_dataset(dance_model, three_dataset)
    mean_loss['checkpoint'] = chk
    with open(f'/nfs/user/Users/ch3849/ProDance/model/{model_date}/evaluation_maxlen_{max_len}.json', 'a') as f:
        json.dump(mean_loss, f)
        f.write('\n')