import torch
import torch.nn as nn
from config import config
from transformers import EsmModel

class ESMwrap(nn.Module):
    def __init__(self, esm2_select):
        super(ESMwrap, self).__init__()
        # Initialize the ESM2 model (35M parameter model)
        self.esm2 = EsmModel.from_pretrained(config[esm2_select]['model_id'])

        # Compute intermediate dimension for pairwise feature prediction based on the input and output feature sizes
        d_pair_middle = int(pow(config[esm2_select]['res_in_feature'] * config[esm2_select]['pair_out_feature'], 1/2))

        # Linear layer for residue-level feature prediction
        self.res_pred_linear = nn.Linear(config[esm2_select]['res_in_feature'], config[esm2_select]['res_out_feature']) 
        
        # Two linear layers for intermediate pairwise feature prediction (one for each residue in the pair)
        self.pair_median_linear_1 = nn.Linear(config[esm2_select]['res_in_feature'], d_pair_middle)
        self.pair_median_linear_2 = nn.Linear(config[esm2_select]['res_in_feature'], d_pair_middle)

        # Final linear layer for pairwise feature prediction (combines pairwise attention and middle features)
        self.pair_pred_linear = nn.Linear(config[esm2_select]['pair_in_feature'] + d_pair_middle, config[esm2_select]['pair_out_feature'])

        # ReLU activation function
        self.relu = nn.ReLU()

    def forward(self, inputs, return_res_emb=False, return_attention_map=False, return_pair_middle=False, return_res_pred=True, return_pair_pred=True):
        # Initialize a dictionary to store outputs
        output = {}

        # Forward pass through the ESM2 model, outputs include attention maps
        esm_output = self.esm2(**inputs, output_attentions=True) 

        # Get the residue embeddings from the last hidden state of the ESM2 model
        res_emb = esm_output['last_hidden_state']

        # Concatenate the attention maps (for all attention heads) and permute the dimensions
        pair_atten = torch.cat(esm_output['attentions'], dim=1).permute(0, 2, 3, 1)

        # Optionally return residue embeddings
        if return_res_emb:
            output['res_emb'] = res_emb

        # Optionally return the attention map for pairwise features
        if return_attention_map:
            output['attention_map'] = pair_atten

        # Residue-level prediction using a linear layer applied to residue embeddings
        res_pred = self.res_pred_linear(res_emb)
        if return_res_pred:
            output['res_pred'] = res_pred

        # Pairwise-level middle features computation using residue embeddings
        pair_middle = self.relu(self.pair_median_linear_1(res_emb)).unsqueeze(2) + self.relu(self.pair_median_linear_2(res_emb)).unsqueeze(1)

        # Optionally return the intermediate pairwise features
        if return_pair_middle:
            output['pair_middle'] = pair_middle

        # Pairwise-level prediction using both pairwise attention and intermediate features
        pair_pred = self.pair_pred_linear(torch.cat([pair_middle, pair_atten], dim=-1))  # Concatenate middle features and attention map

        # Optionally return the final pairwise prediction
        if return_pair_pred:
            output['pair_pred'] = pair_pred

        # Return the output dictionary containing requested results
        return output