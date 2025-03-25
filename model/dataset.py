import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from transformers import AutoTokenizer
from config import config

def collate_batch(batch):
    batch_collated = {}
    batch_collated['input_ids'] = pad_sequence([b['input_ids'] for b in batch], batch_first=True, padding_value=1) # 1 is <pad> in tokenization
    batch_collated['attention_mask'] = pad_sequence([b['attention_mask'] for b in batch], batch_first=True, padding_value=0) # 0 is to be masked
    batch_collated['res_feat'] = pad_sequence([b['res_feat'] for b in batch], batch_first=True, padding_value=-1) # -1 is to be masked in the calculation of loss

    max_len = batch_collated['input_ids'].shape[1]
    pair_padded = [F.pad(b['pair_feat'], (0,0,0, max_len-b['pair_feat'].size(0), 0, max_len-b['pair_feat'].size(0)), mode='constant', value=-1) for b in batch]
    batch_collated['pair_feat'] = torch.stack(pair_padded)

    return batch_collated

class ProteinDataset(Dataset):
    def __init__(self, df, h5py_read, esm2_select, max_len, start_from_N_terminal=False):
        self.names = list(df['name'])
        self.seqs = list(df['modify_seq'])
        self.h5py_read = h5py_read
        self.tokenizer = AutoTokenizer.from_pretrained(config[esm2_select]['model_id'])
        self.max_len = max_len
        self.start_from_N_terminal = start_from_N_terminal

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if isinstance(idx, list):
            batch = [self._process_single_item(i) for i in idx]
            return batch
        else:
            return self._process_single_item(idx)

    def _process_single_item(self, idx):
        name = self.names[idx]
        seq = self.seqs[idx]

        res_feat = torch.tensor(self.h5py_read[f'{name}_res_feature'][:])
        pair_feat = torch.tensor(self.h5py_read[f'{name}_pair_feature'][:])

        # normalize the rmsf feature by the max value of the protein
        # only for ATLAS_GPCRmd_PED_mdCATH and IDRome which have 47 features
        if res_feat.shape[1] == 47:
            rmsf_feat = res_feat[:, 2]
            # Identify non-padding values (values not equal to -1)
            valid_mask = rmsf_feat != -1  
            # Normalize only valid values
            rmsf_feat[valid_mask] = rmsf_feat[valid_mask] / rmsf_feat[valid_mask].max()
            # Assign back to the original array
            res_feat[:, 2] = rmsf_feat

        raw_input = self.tokenizer(seq, return_tensors="pt")
        item = {key: val.squeeze(0) for key, val in raw_input.items()}
        length = item['input_ids'].shape[0]

        if length > self.max_len:
            if self.start_from_N_terminal:
                start = 0
            else:
                start = torch.randint(0, length - self.max_len + 1, (1,)).item()
            item['input_ids'] = item['input_ids'][start:start + self.max_len]
            item['attention_mask'] = item['attention_mask'][start:start + self.max_len]
            res_feat = res_feat[start:start + self.max_len, :]
            pair_feat = pair_feat[start:start + self.max_len, start:start + self.max_len, :]

        item['res_feat'] = res_feat
        item['pair_feat'] = pair_feat
        
        return item
