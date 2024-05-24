"""
Code based on: https://github.com/google-research/rigl
"""

import torch
import logging
import numpy as np

from torch import nn

def ERK(W, density, erk_power_scale: float = 1.0):
    total_params = 0
    for weight in W:
        total_params += weight.numel()

    is_epsilon_valid = False
    dense_layers = set()
    while not is_epsilon_valid:
        divisor = 0
        rhs = 0
        raw_probabilities = [None for _ in W]
        for idx, weight in enumerate(W):
            n_param = np.prod(weight.shape)
            n_zeros = n_param * (1 - density)
            n_ones = n_param * density

            if idx in dense_layers:
                rhs -= n_zeros

            else:
                rhs += n_ones
                raw_probabilities[idx] = (
                    np.sum(weight.shape) / np.prod(weight.shape)
                ) ** erk_power_scale
                divisor += raw_probabilities[idx] * n_param
        epsilon = rhs / divisor
        max_prob = np.max([x for x in raw_probabilities if x is not None])
        max_prob_one = max_prob * epsilon
        if max_prob_one > 1:
            is_epsilon_valid = False
            for idx, mask_raw_prob in enumerate(raw_probabilities):
                if mask_raw_prob == max_prob:
                    logging.info(f"Sparsity of var:{idx} had to be set to 0.")
                    dense_layers.add(idx)
        else:
            is_epsilon_valid = True

    density_dict = [None for _ in W]
    total_nonzero = 0.0
    for idx, weight in enumerate(W):
        n_param = np.prod(weight.shape)
        if idx in dense_layers:
            density_dict[idx] = 1.0
        else:
            probability_one = epsilon * raw_probabilities[idx]
            density_dict[idx] = probability_one
        logging.info(
            f"layer: {idx}, shape: {weight.shape}, density: {density_dict[idx]}"
        )
        total_nonzero += density_dict[idx] * weight.numel()
    logging.info(f"Overall sparsity {total_nonzero/total_params}")
    return density_dict