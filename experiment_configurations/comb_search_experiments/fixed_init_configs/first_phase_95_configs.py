import math

first_phase_95_experiments = []

epochs = 50
lr_list = [0.05, 0.1]
weight_decay_list = [5e-4]
momentum_list = [0.9]
batch_size_list = [128]
width_list = [16]
prune_type_list = ["comb_search"]
lr_milestones_list = ["cosine_annealing", [0.3, 0.6], []]
d1_list = [1,2,3,4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
d2_list = [1,2,3,4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
d3_list = [1,2,3,4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
out_dim = 1
in_dim = 2

bench_nnz = (2*3)+(3*3)+(3*3)+(3*1)+(3+3+3+1)

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
                                        curr_nnz = (2*d1_list[d1_idx]) + (d1_list[d1_idx]*d2_list[d2_idx])+(d2_list[d2_idx]*d3_list[d3_idx])+(d3_list[d3_idx]*1)+(d1_list[d1_idx]+d2_list[d2_idx]+d3_list[d3_idx]+1)
                                        if curr_nnz < bench_nnz:
                                            first_phase_95_experiments.append(
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
                                                    'w1_nnz'            : "NA",
                                                    'w1_mask'           : "NA",
                                                    'w2_nnz'            : "NA",
                                                    'w2_mask'           : "NA",
                                                    'w3_nnz'            : "NA",
                                                    'w3_mask'           : "NA",
                                                    'clf_nnz'           : "NA",
                                                    'clf_mask'          : "NA",
                                                    'out_dim'           : out_dim,
                                                    'in_dim'            : in_dim,
                                                    'width'             : width_list[w_idx]
                                                }
                                            )