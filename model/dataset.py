import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from transformers import AutoTokenizer
from config import config

# Function to collate a batch of samples
def collate_batch(batch):
    batch_collated = {}
    
    # Pad the input token IDs to the same length (padding value 1 corresponds to <pad> token)
    batch_collated['input_ids'] = pad_sequence(
        [b['input_ids'] for b in batch], batch_first=True, padding_value=1
    )

    # Pad the attention mask (padding value 0 means these positions are ignored in attention computation)
    batch_collated['attention_mask'] = pad_sequence(
        [b['attention_mask'] for b in batch], batch_first=True, padding_value=0
    )

    # Pad residue-level features (padding value -1 indicates masked values in loss calculations)
    batch_collated['res_feat'] = pad_sequence(
        [b['res_feat'] for b in batch], batch_first=True, padding_value=-1
    )

    # Determine the maximum sequence length in the batch
    max_len = batch_collated['input_ids'].shape[1]

    # Pad pairwise features to match the maximum sequence length in both dimensions
    pair_padded = [
        F.pad(
            b['pair_feat'],
            (0, 0, 0, max_len - b['pair_feat'].size(0), 0, max_len - b['pair_feat'].size(0)), 
            mode='constant', 
            value=-1  # Padding value -1 for masked positions
        ) 
        for b in batch
    ]
    
    # Stack the padded pairwise feature tensors
    batch_collated['pair_feat'] = torch.stack(pair_padded)

    return batch_collated

# Custom dataset class for handling protein sequences and features
class ProteinDataset(Dataset):
    def __init__(self, df, h5py_read, esm2_select, max_len, start_from_N_terminal=False):
        """
        Initializes the dataset with sequence names, modified sequences, and feature storage.
        
        Args:
        - df: Pandas DataFrame containing protein names and modified sequences.
        - h5py_read: HDF5 file handler for reading precomputed dynamic features.
        - esm2_select: Key to select the appropriate ESM model.
        - max_len: Maximum allowed sequence length.
        - start_from_N_terminal: If True, always start tokenization from the N-terminal.
        """
        self.names = list(df['name'])  # List of protein names
        self.seqs = list(df['modify_seq'])  # List of modified protein sequences
        self.h5py_read = h5py_read  # HDF5 dataset reader
        self.tokenizer = AutoTokenizer.from_pretrained(config[esm2_select]['model_id'])  # Load tokenizer
        self.max_len = max_len  # Maximum sequence length
        self.start_from_N_terminal = start_from_N_terminal  # Flag to control sequence truncation behavior

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.names)

    def __getitem__(self, idx):
        """
        Retrieves a single item or a batch from the dataset.

        Args:
        - idx: Integer index or a list of indices.

        Returns:
        - Single sample dictionary if idx is an integer.
        - List of sample dictionaries if idx is a list.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()  # Convert tensor index to list if necessary

        if isinstance(idx, list):
            batch = [self._process_single_item(i) for i in idx]  # Process multiple items for batch
            return batch
        else:
            return self._process_single_item(idx)  # Process a single item

    def _process_single_item(self, idx):
        """
        Processes a single protein sequence and its associated features.

        Args:
        - idx: Integer index corresponding to a protein.

        Returns:
        - Dictionary containing tokenized sequence, attention mask, and extracted features.
        """
        name = self.names[idx]  # Get protein name
        seq = self.seqs[idx]  # Get modified sequence

        # Load residue-level features and pairwise interaction features from HDF5 dataset
        res_feat = torch.tensor(self.h5py_read[f'{name}_res_feature'][:])
        pair_feat = torch.tensor(self.h5py_read[f'{name}_pair_feature'][:])

        # Normalize RMSF feature (Root Mean Square Fluctuation) for specific datasets
        if res_feat.shape[1] == 47:  # Condition: Only normalize for datasets with 47 features
            rmsf_feat = res_feat[:, 2]  # Extract the RMSF column (assumed to be at index 2)
            
            # Identify non-padding values (i.e., values not equal to -1)
            valid_mask = rmsf_feat != -1  
            
            # Normalize only valid RMSF values by dividing by the maximum value
            rmsf_feat[valid_mask] = rmsf_feat[valid_mask] / rmsf_feat[valid_mask].max()
            
            # Assign the normalized values back to the original feature matrix
            res_feat[:, 2] = rmsf_feat

        # Tokenize the protein sequence using the selected model's tokenizer
        raw_input = self.tokenizer(seq, return_tensors="pt")
        
        # Convert tokenized tensors from batch format (1, seq_length) to (seq_length)
        item = {key: val.squeeze(0) for key, val in raw_input.items()}

        # Get sequence length
        length = item['input_ids'].shape[0]

        # Truncate the sequence if it exceeds max_len
        if length > self.max_len:
            if self.start_from_N_terminal:
                start = 0  # Always start from the N-terminal
            else:
                start = torch.randint(0, length - self.max_len + 1, (1,)).item()  # Random start position
            
            # Apply truncation to input IDs and attention mask
            item['input_ids'] = item['input_ids'][start:start + self.max_len]
            item['attention_mask'] = item['attention_mask'][start:start + self.max_len]

            # Apply truncation to residue-level and pairwise features
            res_feat = res_feat[start:start + self.max_len, :]
            pair_feat = pair_feat[start:start + self.max_len, start:start + self.max_len, :]

        # Store processed features in the item dictionary
        item['res_feat'] = res_feat
        item['pair_feat'] = pair_feat
        
        return item
