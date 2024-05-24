import math 
from itertools import combinations
import numpy as np
import torch
from pruning_methods.Mask import Mask


def get_search_mask(model, list_of_dim, list_of_mask):
    scores = {}
    for weight_idx, weight in enumerate(model.weight_params):
        key = "weight_param_" + str(weight_idx)
        curr_score_w = torch.zeros_like(weight.data.clone()).cpu().detach().numpy()
        if list_of_mask[weight_idx]:
            curr_score_w[:list_of_dim[weight_idx+1], :list_of_dim[weight_idx]] = list_of_mask[weight_idx]
        else:
            curr_score_w[:list_of_dim[weight_idx+1], :list_of_dim[weight_idx]] = 1.0
        scores[key] = curr_score_w
    
    for bias_idx, bias in enumerate(model.bias_params):
        key = "bias_param_" + str(bias_idx)
        curr_score_b = torch.zeros_like(bias.data.clone()).cpu().detach().numpy()
        curr_score_b[:list_of_dim[bias_idx+1]] = 1.0
        scores[key] = curr_score_b
    
    supp_dict = {k: np.where(np.abs(v) > 0.5, np.ones_like(v), np.zeros_like(v))
                        for k, v in scores.items()}
    new_mask = Mask(supp_dict)
    return new_mask

def get_num_of_supp(mid_dim_one, mid_dim_two, nnz_count):
    assert nnz_count >= max([mid_dim_one, mid_dim_two])
    assert nnz_count <= mid_dim_one*mid_dim_two
    all_supp = get_all_valid_supp(mid_dim_one, mid_dim_two, nnz_count)
    return(len(all_supp))

def get_sub_mask(mid_dim_one, mid_dim_two, nnz_count, mask_config):
    if mask_config == "NA":
        return None
    else:
        all_supp = get_all_valid_supp(mid_dim_one, mid_dim_two, nnz_count)
        return all_supp[mask_config]

def get_all_valid_supp(mid_dim_one, mid_dim_two, nnz_count):
    assert nnz_count >= max([mid_dim_one, mid_dim_two])
    assert nnz_count <= mid_dim_one*mid_dim_two

    list_of_supp = []
    list_of_budgets = recursive_construct_budget(mid_dim_one, mid_dim_two, nnz_count)
    for budget in list_of_budgets:
        top_row_budget = budget[0]
        list_of_top_rows= [ [1 if i in comb else 0 for i in range(mid_dim_one)]
                for comb in combinations(np.arange(mid_dim_one), top_row_budget)]
        duplicate=False
        if len(budget) > 1:
            if top_row_budget == budget[1]:
                duplicate = True
            # need to check if there is zero column when adding the final row
            rest_of_mats = recursive_construct_supp(mid_dim_one, mid_dim_two-1, budget[1:])
            for lower_mat in rest_of_mats:
                for row in list_of_top_rows:
                    viable_config = True
                    for col_idx in range(len(row)):
                        if row[col_idx] == 0:
                            deadCol = True
                            for lower_row in lower_mat:
                                if lower_row[col_idx] != 0:
                                    deadCol = False
                            if deadCol:
                                viable_config = False
                    # check binary condition
                    if duplicate:
                        curr_row_bin = convert_to_binary(row)
                        next_row_bin = convert_to_binary(lower_mat[0])
                        if curr_row_bin < next_row_bin:
                            viable_config = False
                    if viable_config:
                        new_supp = [row] + lower_mat
                        list_of_supp.append(new_supp)
        else:
            for top_row in list_of_top_rows:
                list_of_supp.append([top_row])
    return list_of_supp

def recursive_construct_budget(mid_dim_one, mid_dim_two, nnz_count):
    if mid_dim_two > 1:
        new_configs = []
        for k in range(math.ceil(nnz_count/mid_dim_two), min([mid_dim_one, nnz_count - (mid_dim_two -1)])+1):
            list_of_configs = recursive_construct_budget(k, mid_dim_two-1, nnz_count-k)
            for config in list_of_configs:
                config.insert(0, k)
                new_configs.append(config)
        return new_configs
    else:
        return [[nnz_count]]

def recursive_construct_supp(mid_dim_one, mid_dim_two, budgets_per_row):
    if mid_dim_two > 1:
        curr_budget = budgets_per_row[0]
        duplicate=False
        if curr_budget == budgets_per_row[1]:
            duplicate = True
        list_of_matrices = []
        list_of_rows= [ [1 if i in comb else 0 for i in range(mid_dim_one)]
                for comb in combinations(np.arange(mid_dim_one), curr_budget)]
        rows_below = recursive_construct_supp(mid_dim_one, mid_dim_two-1, budgets_per_row[1:])
        for below_matrix in rows_below:
            for row in list_of_rows:
                if duplicate:
                    next_row = below_matrix[0]
                    next_row_bin = convert_to_binary(next_row)
                    curr_row_bin = convert_to_binary(row)
                    if curr_row_bin >= next_row_bin:
                        new_mat = [row] + below_matrix
                        list_of_matrices.append(new_mat)
                else:
                    new_mat = [row] + below_matrix
                    list_of_matrices.append(new_mat)
        return list_of_matrices
    else:
        return [ [[1 if i in comb else 0 for i in range(mid_dim_one)]]
                for comb in combinations(np.arange(mid_dim_one), budgets_per_row[0])]

def convert_to_binary(row):
    num = 0
    for b in row:
        num = 2 * num + b
    return num