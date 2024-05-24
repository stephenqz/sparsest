import math
from combinatorial_search.combinatorial_search import get_num_of_supp

second_phase_995_exper = []

epochs = 50
lr_list = [0.05]
weight_decay_list = [5e-4]
momentum_list = [0.9]
batch_size_list = [128]

width_list = [16]
prune_type_list = ["comb_search"]
lr_milestones_list = ["cosine_annealing", [0.3, 0.6], []]
d1_list = [6]
d2_list = [3]
d3_list = [2]
out_dim = 1
in_dim = 2

for lr_idx in range(len(lr_list)):
    for bs_idx in range(len(batch_size_list)):
        for lrML_idx in range(len(lr_milestones_list)):
            for mom_idx in range(len(momentum_list)):
                for pt_idx in range(len(prune_type_list)):
                    for wd_idx in range(len(weight_decay_list)):
                        epoch_percents = lr_milestones_list[lrML_idx]
                        if epoch_percents != "cosine_annealing":
                            lr_milestones = [math.ceil(epochs*curr_percent) for curr_percent in epoch_percents]
                        else:
                            lr_milestones = "cosine_annealing"
                        for w_idx in range(len(width_list)):
                            for d1_idx in range(len(d1_list)):
                                for d2_idx in range(len(d2_list)):
                                    for d3_idx in range(len(d3_list)):
                                        w1_nnz_list = [i for i in range(max([in_dim, d1_list[d1_idx]]), in_dim*d1_list[d1_idx]+1)]
                                        w2_nnz_list = [i for i in range(max([d2_list[d2_idx],d1_list[d1_idx]]), d1_list[d1_idx]*d2_list[d2_idx]+1)]
                                        w3_nnz_list = [i for i in range(max([d3_list[d3_idx],d2_list[d2_idx]]), d3_list[d3_idx]*d2_list[d2_idx]+1)]

                                        w1_mask_dict = {}
                                        w2_mask_dict = {}
                                        w3_mask_dict = {}

                                        for w1nz_idx in range(len(w1_nnz_list)):
                                            w1_mask_dict[w1_nnz_list[w1nz_idx]] = get_num_of_supp(in_dim, d1_list[d1_idx], w1_nnz_list[w1nz_idx])
                                        for w2nz_idx in range(len(w2_nnz_list)):
                                            w2_mask_dict[w2_nnz_list[w2nz_idx]] = get_num_of_supp(d1_list[d1_idx], d2_list[d2_idx], w2_nnz_list[w2nz_idx])
                                        for w3nz_idx in range(len(w3_nnz_list)):
                                            w3_mask_dict[w3_nnz_list[w3nz_idx]] = get_num_of_supp(d2_list[d2_idx], d3_list[d3_idx], w3_nnz_list[w3nz_idx])

                                        for w1nz_idx in range(len(w1_nnz_list)):
                                            for w2nz_idx in range(len(w2_nnz_list)):
                                                for w3nz_idx in range(len(w3_nnz_list)):
                                                    if w1_nnz_list[w1nz_idx] + w2_nnz_list[w2nz_idx] + w3_nnz_list[w3nz_idx] < 31:
                                                        w1_mask_list = [i for i in range(w1_mask_dict[w1_nnz_list[w1nz_idx]])]
                                                        w2_mask_list = [i for i in range(min([w2_mask_dict[w2_nnz_list[w2nz_idx]], 6]))]
                                                        w3_mask_list = [i for i in range(w3_mask_dict[w3_nnz_list[w3nz_idx]])]
                                                        for w1m_idx in range(len(w1_mask_list)):
                                                            for w2m_idx in range(len(w2_mask_list)):
                                                                for w3m_idx in range(len(w3_mask_list)):
                                                                    second_phase_995_exper.append(
                                                                    {
                                                                        'lr_milestones'     : lr_milestones,
                                                                        'd1'                : d1_list[d1_idx],
                                                                        'd2'                : d2_list[d2_idx],
                                                                        'd3'                : d3_list[d3_idx],
                                                                        'momentum'          : momentum_list[mom_idx],
                                                                        'lr'                : lr_list[lr_idx],
                                                                        'batch_size'        : batch_size_list[bs_idx],
                                                                        'weight_decay'      : weight_decay_list[wd_idx],
                                                                        'epochs'            : epochs,
                                                                        'prune_type'        : prune_type_list[pt_idx],
                                                                        'w1_nnz'            : w1_nnz_list[w1nz_idx],
                                                                        'w1_mask'           : w1_mask_list[w1m_idx],
                                                                        'w2_nnz'            : w2_nnz_list[w2nz_idx],
                                                                        'w2_mask'           : w2_mask_list[w2m_idx],
                                                                        'w3_nnz'            : w3_nnz_list[w3nz_idx],
                                                                        'w3_mask'           : w3_mask_list[w3m_idx],
                                                                        'clf_nnz'           : "NA",
                                                                        'clf_mask'          : "NA",
                                                                        'out_dim'           : out_dim,
                                                                        'in_dim'            : in_dim,
                                                                        'width'             : width_list[w_idx]
                                                                    }
                                                                    )