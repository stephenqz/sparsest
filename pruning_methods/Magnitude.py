"""
Code based on: https://github.com/facebookresearch/open_lth
"""

import numpy as np
from pruning_methods.Mask import Mask

def magnitude_prune(model, final_num_weights):   
    weights = {}
    for weight_idx, weight in enumerate(model.weight_params):
        key = "weight_param_" + str(weight_idx)
        weights[key] = weight.data.clone().cpu().detach().numpy()

    biases = {}
    for bias_idx, bias in enumerate(model.bias_params):
        key = "bias_param_" + str(bias_idx)
        biases[key] = bias.data.clone().cpu().detach().numpy()

    threshold_params = weights
    param_vector = np.concatenate([v.flatten() for k, v in threshold_params.items()])

    N = len(param_vector)
    weights_to_prune = N-final_num_weights

    if weights_to_prune >= 1:
        threshold = np.sort(np.abs(param_vector))[weights_to_prune]
        supp_dict = {k: np.where(np.abs(v) > threshold, np.ones_like(v), np.zeros_like(v))
                        for k, v in threshold_params.items()}
        # mask biases
        bias_supp = {}
        for bias_idx, bias in enumerate(model.bias_params):
            key = "bias_param_" + str(bias_idx)
            if bias_idx < len(model.bias_params) - 1:
                curr_supp_w = supp_dict["weight_param_"+str(bias_idx+1)]
                bias_supp[key] = np.sign(np.count_nonzero(curr_supp_w, axis=0))
            else:
                bias_supp[key] = np.ones_like(biases[key])
        supp_dict.update(bias_supp)
        new_mask = Mask(supp_dict)
    else:
        new_mask = Mask.ones_like(model)

    return new_mask