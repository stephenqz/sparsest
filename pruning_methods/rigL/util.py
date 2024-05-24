""" 
Code based on: https://github.com/verbocado/rigl-torch
"""

import torch
import torchvision

def get_weighted_layers(model):
    layers = []
    linear_layers_mask = []
    items = model._modules.items()
    for layer_name, p in items:
        if isinstance(p, torch.nn.Linear):
            layers.append([p])
            linear_layers_mask.append(1)
    return layers, linear_layers_mask 



def get_W(model):
    layers, linear_layers_mask = get_weighted_layers(model)
    W = []
    for layer in layers:
        W.append(layer[0].weight)
    assert len(W) == len(linear_layers_mask)
    return W

def get_B(model):
    layers, linear_layers_mask = get_weighted_layers(model)
    B = []
    for layer in layers:
        B.append(layer[0].bias)
    assert len(B) == len(linear_layers_mask)
    return B