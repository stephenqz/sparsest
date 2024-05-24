"""
Code based on: https://github.com/naver/force
"""

import torch
import torch.nn as nn
import copy
import numpy as np

import warnings


####################################################
############### Get saliencies    ##################
####################################################

def get_average_gradients(net, train_dataloader, criterion, device, num_batches=-1):
    """
    Function to compute gradients and average them over several batches.
    
    num_batches: Number of batches to be used to approximate the gradients. 
                 When set to -1, uses the whole training set.
    
    Returns a list of tensors, with gradients for each prunable layer.
    """
    
    # Prepare list to store gradients
    gradients = []
    for layer in net.modules():
        # Select only prunable layers
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            gradients.append(0)
    
    # Take a whole epoch
    count_batch = 0
    for batch_idx in range(len(train_dataloader)):
        inputs, targets = next(iter(train_dataloader))
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # Compute gradients (but don't apply them)
        net.zero_grad()
        outputs = net.forward(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        
        # Store gradients
        counter = 0
        for layer in net.modules():
            # Select only prunable layers
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                gradients[counter] += layer.weight.grad
                counter += 1
        count_batch += 1
        if batch_idx == num_batches - 1:
            break
    avg_gradients = [x / count_batch for x in gradients] 
        
    return avg_gradients

    
def get_average_saliencies(net, train_dataloader, criterion, device, prune_method=3, num_batches=-1,
                           original_weights=None):
    """
    Get saliencies with averaged gradients.
    
    num_batches: Number of batches to be used to approximate the gradients. 
                 When set to -1, uses the whole training set.
    
    prune_method:
                   1: Use Iter SNIP.
                   2: Use GRASP-It.
                   3: Use FORCE (default).
                   4: Random (random pruning baseline).
    
    Returns a list of tensors with saliencies for each weight.
    """
     
    def pruning_criteria(method):
        if method == 2:
            # GRASP-It method 
            result = layer_weight_grad**2 # Custom gradient norm approximation
        elif method == 4:
            result = torch.rand_like(layer_weight) # Randomly pruning weights
        else:
            # FORCE / Iter SNIP method
            result = torch.abs(layer_weight * layer_weight_grad)
        return result

    if prune_method != 4: # No need to compute gradients for random pruning
        gradients = get_average_gradients(net, train_dataloader, criterion, device, num_batches)
    saliency = []
    idx = 0
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            if prune_method == 3:
                layer_weight = original_weights[idx]
            else:
                layer_weight = layer.weight
            if prune_method != 4: # No need to compute gradients for random pruning
                layer_weight_grad = gradients[idx]
            idx += 1
            saliency.append(pruning_criteria(prune_method))
                
    return saliency
    
###################################################
############# Iterative pruning ###################
###################################################
        
def get_mask(saliency, pruning_factor):
    """ 
    Given a list of saliencies and a pruning factor (sparsity),
    returns a list with binary tensors which correspond to pruning masks.
    """
    all_scores = torch.cat([torch.flatten(x) for x in saliency])

    num_params_to_keep = int(len(all_scores) * pruning_factor)
    threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
    acceptable_score = threshold[-1]

    keep_masks = []
    for m in saliency:
        #keep_masks.append((m >= acceptable_score).float())
        keep_masks.append((m > acceptable_score).float())
    return keep_masks

def iterative_pruning(ori_net, train_dataloader, criterion, device, pruning_factor=0.1,
                      prune_method=3, num_steps=10,
                      mode='exp', num_batches=1):
    """
    Function to gradually remove weights from a network, recomputing the saliency at each step.
    
    pruning_factor: Fraction of remaining weights (globally) after pruning. 
    
    prune_method: Which method to use to prune the layers:
                   1: Use Iter SNIP.
                   2: Use GRASP-It.
                   3: Use FORCE (default).
                   4: Random (random pruning baseline).
                   
    num_steps: Number of iterations to do when pruning progressively (should be >= 1).  
                   
    mode: Mode of choosing the sparsity decay schedule. One of 'exp', 'linear'
                   
    num_batches: Number of batches to be used to approximate the gradients (should be -1 or >= 1). 
                 When set to -1, uses the whole training set.
                 
    Returns a list of binary tensors which correspond to the final pruning mask.
    """
    print('pruning factor = ' + str(pruning_factor))
    net = copy.deepcopy(ori_net)
    
    if prune_method == 4 and num_steps > 1:
        message = 'The selected pruning variant (Random) is not meant to perform iterative pruning'
        warnings.warn(message, UserWarning, stacklevel=2)
    
    if prune_method == 3:
        original_weights = []
        for layer in net.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                original_weights.append(layer.weight.detach())
    else:
        original_weights = None
        
    if mode == 'linear':
        pruning_steps = [1 - ((x + 1) * (1 - pruning_factor) / num_steps) for x in range(num_steps)]
        
    elif mode == 'exp':
        pruning_steps = [np.exp(0 - ((x + 1) * (0 - np.log(pruning_factor)) / num_steps)) for x in range(num_steps)]
   
    mask = None
    hook_handlers = None
    
    for perc in pruning_steps:
        saliency = []
        saliency = get_average_saliencies(net, train_dataloader, criterion, device,
                                          prune_method=prune_method,
                                          num_batches=num_batches,
                                          original_weights=original_weights)
        torch.cuda.empty_cache()
        
        if mask is not None and prune_method < 3:
            min_saliency = get_minimum_saliency(saliency)
            for ii in range(len(saliency)):
                saliency[ii][mask[ii] == 0.] = min_saliency
        
        if hook_handlers is not None:
            for h in hook_handlers:
                h.remove()
                
        mask = []
        mask = get_mask(saliency, perc)
        
        if prune_method == 3:
            net = copy.deepcopy(ori_net)
            apply_prune_mask(net, mask, apply_hooks=False)
        else:   
            hook_handlers = apply_prune_mask(net, mask, apply_hooks=True)
        
        p = check_global_pruning(mask)
        print(f'Global pruning {round(float(p),5)}')
    
    return mask 

def check_global_pruning(mask):
    "Compute fraction of unpruned weights in a mask"
    flattened_mask = torch.cat([torch.flatten(x) for x in mask])
    return flattened_mask.mean()

def get_minimum_saliency(saliency):
    "Compute minimum value of saliency globally"
    flattened_saliency = torch.cat([torch.flatten(x) for x in saliency])
    return flattened_saliency.min()

def get_maximum_saliency(saliency):
    "Compute maximum value of saliency globally"
    flattened_saliency = torch.cat([torch.flatten(x) for x in saliency])
    return flattened_saliency.max()


####################################################################
######################    UTILS    #################################
####################################################################

def get_force_saliency(net, mask, train_dataloader, criterion, device, num_batches):
    """
    Given a dense network and a pruning mask, compute the FORCE saliency.
    """
    net = copy.deepcopy(net)
    apply_prune_mask(net, mask, 0, apply_hooks=True)
    saliencies = get_average_saliencies(net, train_dataloader, criterion, device,
                                        1, num_batches=num_batches)
    torch.cuda.empty_cache()
    s = sum_unmasked_saliency(saliencies, mask)
    torch.cuda.empty_cache()
    return s

def sum_unmasked_saliency(variable, mask):
    "Util to sum all unmasked (mask==1) components"
    V = 0
    for v, m in zip(variable, mask):
        V += v[m > 0].sum()
    return V.detach().cpu()

def get_gradient_norm(net, mask, train_dataloader, criterion, device, num_batches):
    "Given a dense network, compute the gradient norm after applying the pruning mask."
    net = copy.deepcopy(net)
    apply_prune_mask(net, mask)
    gradients = get_average_gradients(net, train_dataloader, criterion, device, num_batches)
    torch.cuda.empty_cache()
    norm = 0
    for g in gradients:
        norm += (g**2).sum().detach().cpu().numpy()
    return norm

def apply_prune_mask(net, keep_masks, apply_hooks=True):
    """
    Function that takes a network and a list of masks and applies it to the relevant layers.
    mask[i] == 0 --> Prune parameter
    mask[i] == 1 --> Keep parameter
    
    If apply_hooks == False, then set weight to 0 but do not block the gradient.
    This is used for FORCE algorithm that sparsifies the net instead of pruning.
    """
    
    prunable_layers = filter(
        lambda layer: isinstance(layer, nn.Conv2d) or isinstance(
            layer, nn.Linear), net.modules())

    hook_handlers = []
    for layer, keep_mask in zip(prunable_layers, keep_masks):
        assert (layer.weight.shape == keep_mask.shape)
        
        def hook_factory(keep_mask):
            """
            The hook function can't be defined directly here because of Python's
            late binding which would result in all hooks getting the very last
            mask! Getting it through another function forces early binding.
            """
            def hook(grads):
                return grads * keep_mask

            return hook

        layer.weight.data[keep_mask == 0.] = 0.
        
        if apply_hooks:
            hook_handlers.append(layer.weight.register_hook(hook_factory(keep_mask)))
        
    return hook_handlers