"""
Code based on: https://github.com/facebookresearch/open_lth
"""

import numpy as np
import torch

class Mask(dict):
    def __init__(self, other_dict=None):
        super(Mask, self).__init__()

        if other_dict is not None:
            for k, v in other_dict.items(): self[k] = v

    def __setitem__(self, key, value):
        if not isinstance(key, str) or len(key) == 0:
            raise ValueError('Invalid tensor name: {}'.format(key))
        if isinstance(value, np.ndarray):
            value = torch.as_tensor(value)
        if not isinstance(value, torch.Tensor):
            raise ValueError('value for key {} must be torch Tensor or numpy ndarray.'.format(key))
        if ((value != 0) & (value != 1)).any(): raise ValueError('All entries must be 0 or 1.')

        super(Mask, self).__setitem__(key, value)

    @staticmethod
    def ones_like(model):
        mask = Mask()
        
        for weight_idx, weight in enumerate(model.weight_params):
            key = "weight_param_" + str(weight_idx)
            mask[key] = torch.ones(list(weight.shape))
        
        for bias_idx, bias in enumerate(model.bias_params):
            key = "bias_param_" + str(bias_idx)
            mask[key] = torch.ones(list(bias.shape))

        return mask

    def numpy(self):
        return {k: v.cpu().numpy() for k, v in self.items()}