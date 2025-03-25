import math
import torch
from torch import nn
import torch.nn.functional as F
from transformers import EsmModel
from config import config
from huggingface_hub import PyTorchModelHubMixin


class ESMwrap(nn.Module, PyTorchModelHubMixin):
    def __init__(self, esm2_select, model_select):
        super().__init__()
        # Load the ESM2 model
        self.esm2 = EsmModel.from_pretrained(config[esm2_select]['model_id'])
        self.freeze_esm = config[model_select]['freeze_esm']

        # Freeze self.esm2 parameters if freeze_esm is True
        if self.freeze_esm:
            for param in self.esm2.parameters():
                param.requires_grad = False
            self.esm2.eval()  # Set to evaluation mode

        # Randomize self.esm2 parameters if randomize_esm is True
        if config[model_select]['randomize_esm']:
            self.randomize_model(self.esm2)

        # dimensions of input and output
        embed_dim = config[esm2_select]['embed_dim']
        res_out_dim = config[esm2_select]['res_out_dim']
        atten_dim = config[esm2_select]['atten_dim']
        pair_out_dim = config[esm2_select]['pair_out_dim']

        # Residue-level prediction layer
        self.res_pred_nn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
            nn.Dropout(config['training']['dropout']),  # Apply dropout after LayerNorm
            nn.Linear(embed_dim, res_out_dim)
        )

        # transform res embedding for Pairwise prediction
        self.res_transform_nn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
            nn.Dropout(config['training']['dropout']),  # Apply dropout after LayerNorm
            nn.Linear(embed_dim, embed_dim*2)
        )

        # Pairwise prediction layer
        self.pair_middle_linear = nn.Linear(embed_dim*2, atten_dim)
        self.pair_pred_linear = nn.Linear(atten_dim + atten_dim, pair_out_dim)

        # Activation functions
        self.gelu = nn.GELU()
        self.softplus = nn.Softplus(beta=1.0, threshold=2.0)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)

        # Feature indices from config
        self.res_feature_idx = config['training']['res_feature_idx']
        self.pair_feature_idx = config['training']['pair_feature_idx']

        # Initialize biases to zero
        self._init_bias_zero()

    def randomize_model(self, model):
        """ Randomize the parameters of the given model. """
        for module_ in model.named_modules():
            if isinstance(module_[1], (torch.nn.Linear, torch.nn.Embedding)):
                if hasattr(module_[1], 'bias') and module_[1].bias is not None:
                    module_[1].bias.data.zero_()
                if hasattr(module_[1], 'weight'):
                    if 'query' in module_[0] or 'key' in module_[0] or 'value' in module_[0]:
                        module_[1].weight = nn.init.xavier_uniform_(module_[1].weight, gain=1 / math.sqrt(2))
                    else:
                        module_[1].weight = nn.init.xavier_uniform_(module_[1].weight)
                            
            elif isinstance(module_[1], nn.LayerNorm):
                if hasattr(module_[1], 'bias'):
                    module_[1].bias.data.zero_()
                if hasattr(module_[1], 'weight'):
                    module_[1].weight.data.fill_(1.0)
                
            elif isinstance(module_[1], nn.Dropout):
                module_[1].p = config['training']['dropout']


    def _init_bias_zero(self):
        """ Set all biases in the model (excluding esm2) to zero. """
        for name, module in self.named_modules():
            if "esm2" not in name and isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)


    def forward(self, inputs, return_res_emb=False, return_attention_map=False, return_res_pred=True, return_pair_pred=True):
        output = {}

        # ESM forward pass, Ensure no gradients are stored for frozen ESM2
        if self.freeze_esm:
            with torch.no_grad():
                esm_output = self.esm2(**inputs, output_attentions=True)
        else:
            esm_output = self.esm2(**inputs, output_attentions=True)

        res_emb = esm_output['last_hidden_state']
        pair_atten = torch.cat(esm_output['attentions'], dim=1).permute(0, 2, 3, 1)

        if return_res_emb:
            output['res_emb'] = res_emb
        if return_attention_map:
            output['attention_map'] = pair_atten

        # Residue-level prediction
        if return_res_pred:
            res_pred = self.res_pred_nn(res_emb)
            for feature in self.res_feature_idx:
                if feature == 'rmsf_nor':
                    # Normalized RMSF (max = 1)
                    output[feature] = self.sigmoid(res_pred[:, :, self.res_feature_idx[feature]])
                elif feature in ['ss', 'chi', 'phi', 'psi']:
                    # Secondary structure, chi, phi, psi sum up to 1
                    output[feature] = self.softmax(res_pred[:, :, self.res_feature_idx[feature]])
                else:
                    # All other features are non-negative
                    output[feature] = self.softplus(res_pred[:, :, self.res_feature_idx[feature]])

        # Pairwise transformation
        s = self.res_transform_nn(res_emb)
        q, k = s.chunk(2, dim=-1)
        prod = q[:, None, :, :] * k[:, :, None, :]
        diff = q[:, None, :, :] - k[:, :, None, :]
        pair_middle = self.gelu(self.pair_middle_linear(torch.cat([prod, diff], dim=-1)))

        # Pairwise prediction
        if return_pair_pred:
            pair_pred = self.pair_pred_linear(torch.cat([pair_middle, pair_atten], dim=-1))

            for feature in self.pair_feature_idx:
                if feature in ['corr', 'nma_pair1', 'nma_pair2', 'nma_pair3']:
                    # Co-movement and NMA co-movement correlations: range [-1, 1]
                    output[feature] = self.sigmoid(pair_pred[:, :, :, self.pair_feature_idx[feature]]) * 2 - 1.0
                else:
                    # All interaction features are non-negative
                    output[feature] = self.softplus(pair_pred[:, :, :, self.pair_feature_idx[feature]])

        return output