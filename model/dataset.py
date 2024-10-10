import random
random.seed(0)  # Sets a fixed seed for reproducibility
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence  # Utility to pad sequences to the same length
import torch.nn.functional as F  # Provides a wide range of useful functions for tensors

# Function to prepare a batch for model input
def collate_batch(batch):
    batch_collated = {}
    
    # Pad input sequences with a value of 1 (padding token)
    batch_collated['input_ids'] = pad_sequence(
        [b['input_ids'] for b in batch], batch_first=True, padding_value=1
    )
    
    # Pad attention masks with a value of 0
    batch_collated['attention_mask'] = pad_sequence(
        [b['attention_mask'] for b in batch], batch_first=True, padding_value=0
    )
    
    # Pad residue-level features with a value of -1
    batch_collated['res_feat'] = pad_sequence(
        [b['res_feat'] for b in batch], batch_first=True, padding_value=-1
    )

    # Get the max sequence length after padding input_ids
    max_len = batch_collated['input_ids'].shape[1]

    # Pad pairwise features (e.g., residue-residue interactions) to fit the maximum length
    pair_padded = [
        F.pad(b['pair_feat'], (0, 0, 0, max_len - b['pair_feat'].size(0), 0, max_len - b['pair_feat'].size(0)),
              mode='constant', value=-1)  # Padding with -1 for missing values
        for b in batch
    ]
    
    # Stack all pairwise features into a single tensor
    batch_collated['pair_feat'] = torch.stack(pair_padded)

    return batch_collated  # Return the padded and collated batch for the model

# Custom dataset class to handle protein sequence data
class ProteinDataset(Dataset):
    def __init__(self, df, h5py_read, tokenizer, max_len):
        self.names = list(df['name'])  # List of protein names from the dataframe
        self.seqs = list(df['modify_seq'])  # List of modified sequences, use "<eos><cls>" to separate sequences in complexes
        self.h5py_read = h5py_read  # Handler for reading features from an HDF5 file (~100GB)
        self.tokenizer = tokenizer  # Tokenizer for protein sequences
        self.max_len = max_len  # Maximum sequence length allowed for processing

    def __len__(self):
        return len(self.names)  # Return the number of proteins in the dataset

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if isinstance(idx, list):
            # If multiple indices are provided, process each item in the list
            batch = [self._process_single_item(i) for i in idx]
            return batch
        else:
            # If a single index is provided, process the corresponding item
            return self._process_single_item(idx)

    # Process a single protein item (sequence and its features)
    def _process_single_item(self, idx):
        name = self.names[idx]  # Get the protein name
        seq = self.seqs[idx]  # Get the protein sequence

        # Load residue-level features (e.g., dynamic properties) from HDF5
        res_feat = torch.tensor(self.h5py_read[f'{name}_res_feature'][:])
        
        # Load pairwise features (e.g., interactions between residues)
        pair_feat = torch.tensor(self.h5py_read[f'{name}_pair_feature'][:])

        # Tokenize the sequence into input IDs and attention masks
        raw_input = self.tokenizer(seq, return_tensors="pt")
        item = {key: val.squeeze(0) for key, val in raw_input.items()}  # Squeeze to remove unnecessary dimensions
        length = item['input_ids'].shape[0]  # Get the length of the tokenized sequence

        # If the sequence is longer than the maximum allowed length, randomly select a fragment
        if length > self.max_len:
            start = random.randint(0, length - self.max_len)  # Random starting point for truncation
            item['input_ids'] = item['input_ids'][start:start + self.max_len]
            item['attention_mask'] = item['attention_mask'][start:start + self.max_len]
            res_feat = res_feat[start:start + self.max_len, :]  # Truncate residue features
            pair_feat = pair_feat[start:start + self.max_len, start:start + self.max_len, :]  # Truncate pairwise features

        # Clamp residue and pairwise features to a predefined range [-2, 2]
        item['res_feat'] = torch.clamp(res_feat, min=-2, max=2)
        item['pair_feat'] = torch.clamp(pair_feat, min=-2, max=2)

        return item  # Return the processed item