import random
random.seed(0)
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

def collate_batch(batch):
    batch_collated = {}
    batch_collated['input_ids'] = pad_sequence([b['input_ids'] for b in batch], batch_first=True, padding_value=1)
    batch_collated['attention_mask'] = pad_sequence([b['attention_mask'] for b in batch], batch_first=True, padding_value=0)
    batch_collated['res_feat'] = pad_sequence([b['res_feat'] for b in batch], batch_first=True, padding_value=-1)

    max_len = batch_collated['input_ids'].shape[1]
    pair_padded = [F.pad(b['pair_feat'], (0,0,0, max_len-b['pair_feat'].size(0), 0, max_len-b['pair_feat'].size(0)), mode='constant', value=-1) for b in batch]
    batch_collated['pair_feat'] = torch.stack(pair_padded)

    return batch_collated

class ProteinDataset(Dataset):
    def __init__(self, df, h5py_read, tokenizer, max_len):
        self.names = list(df['name'])
        self.seqs = list(df['modify_seq'])
        self.h5py_read = h5py_read
        self.tokenizer = tokenizer
        self.max_len = max_len

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

        raw_input = self.tokenizer(seq, return_tensors="pt")
        item = {key: val.squeeze(0) for key, val in raw_input.items()}
        length = item['input_ids'].shape[0]

        if length > self.max_len:
            start = random.randint(0, length - self.max_len)
            item['input_ids'] = item['input_ids'][start:start + self.max_len]
            item['attention_mask'] = item['attention_mask'][start:start + self.max_len]
            res_feat = res_feat[start:start + self.max_len, :]
            pair_feat = pair_feat[start:start + self.max_len, start:start + self.max_len, :]

        item['res_feat'] = torch.clamp(res_feat, min=-2, max=2)
        item['pair_feat'] = torch.clamp(pair_feat, min=-2, max=2)

        return item
