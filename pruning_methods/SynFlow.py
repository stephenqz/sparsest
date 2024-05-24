"""
Code based on: https://github.com/ganguli-lab/Synaptic-Flow
"""

import numpy as np
import torch
from pruning_methods.Mask import Mask

def synFlow(model, final_num_weights, inputdim):
    scores = {}
    @torch.no_grad()
    def linearize(model):
        signs = {}
        for weight_idx, weight in enumerate(model.weight_params):
            key = "weight_param_" + str(weight_idx)
            signs[key] = torch.sign(weight)
            weight.abs_()
        
        for bias_idx, bias in enumerate(model.bias_params):
            key = "bias_param_" + str(bias_idx)
            signs[key] = torch.sign(bias)
            bias.abs_()
        
        return signs

    @torch.no_grad()
    def nonlinearize(model, signs):
        for weight_idx, weight in enumerate(model.weight_params):
            key = "weight_param_" + str(weight_idx)
            weight.mul_(signs[key])
        for bias_idx, bias in enumerate(model.bias_params):
            key = "bias_param_" + str(bias_idx)
            bias.mul_(signs[key])

    model.eval()
    for it in range(100):
        signs = linearize(model)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input = torch.ones([1] + inputdim).to(device)
        output = model(input)
        torch.sum(output).backward()
        
        for weight_idx, weight in enumerate(model.weight_params):
            key = "weight_param_" + str(weight_idx)
            scores[key] = (weight.grad*weight).data.abs().clone().cpu().detach().numpy()
            weight.grad.data.zero_()
        nonlinearize(model, signs)
        scores_vector = np.concatenate([v.flatten() for k, v in scores.items()])

        N = len(scores_vector)
        ratio = (it+1.0)/100.0
        curr_weights_to_prune = int(N - ((final_num_weights**(ratio))/(N**(ratio-1.0))))

        if curr_weights_to_prune >= 1: 
            threshold = np.sort(scores_vector)[curr_weights_to_prune]
            supp_dict = Mask({k: np.where(scores[k] > threshold, np.ones_like(v), np.zeros_like(v))
                                for k, v in scores.items()})
            bias_supp = {}
            for bias_idx, bias in enumerate(model.bias_params):
                key = "bias_param_" + str(bias_idx)
                if bias_idx < len(model.bias_params) - 1:
                    curr_supp_w = supp_dict["weight_param_"+str(bias_idx+1)]
                    bias_supp[key] = np.sign(np.count_nonzero(curr_supp_w, axis=0))
                else:
                    bias_supp[key] = np.ones_like(bias.data.clone().cpu().detach().numpy())
            supp_dict.update(bias_supp)

            new_mask = Mask(supp_dict)
        else:
            new_mask = Mask.ones_like(model)
        
        model.update_mask(new_mask)

    return new_mask