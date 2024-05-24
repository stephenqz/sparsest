import math

overparam_prune_exper = []

lr_list = [0.05, 0.1, 0.2]
weight_decay_list = [5e-4]
momentum_list = [0.9]
batch_size_list = [128]

width_list = [3, 4, 5, 6, 7, 8, 16, 32, 64, 128, 256]
prune_type_list = ["imp_LTH", "rigL", "prospr", "iter_snip",  "FORCE", "imp", "grasp", "snip", "synFlow"]
lr_milestones_list =["cosine_annealing", [0.3, 0.6], []]
final_nnz_weights_list = [15, 16, 17, 18, 19,20, 21, 22, 23, 24, 25, 26, 30, 33, 37, 40, 44, 50, 53, 55, 57, 60, 65]
out_dim = 1
in_dim = 2

for lr_idx in range(len(lr_list)):
    for bs_idx in range(len(batch_size_list)):
        for lrML_idx in range(len(lr_milestones_list)):
            for mom_idx in range(len(momentum_list)):
                for pt_idx in range(len(prune_type_list)):
                    for wd_idx in range(len(weight_decay_list)):
                        epoch_percents = lr_milestones_list[lrML_idx]
                        for w_idx in range(len(width_list)):
                            if prune_type_list[pt_idx] == "imp_LTH":
                                epochs = 250
                            else:
                                epochs = 50
                            if epoch_percents != "cosine_annealing":
                                if prune_type_list[pt_idx] == "imp_LTH":
                                    lr_milestones = [math.ceil(50*curr_percent) for curr_percent in epoch_percents]
                                else:
                                    lr_milestones = [math.ceil(epochs*curr_percent) for curr_percent in epoch_percents]
                            else:
                                lr_milestones = "cosine_annealing"
                            for fnw_idx in range(len(final_nnz_weights_list)):
                                curr_width = width_list[w_idx]
                                total_weights = 2*curr_width + curr_width*curr_width + curr_width*curr_width + curr_width
                                if final_nnz_weights_list[fnw_idx] < total_weights:
                                    overparam_prune_exper.append(
                                        {
                                            'lr_milestones'     : lr_milestones,
                                            'momentum'          : momentum_list[mom_idx],
                                            'lr'                : lr_list[lr_idx],
                                            'batch_size'        : batch_size_list[bs_idx],
                                            'weight_decay'      : weight_decay_list[wd_idx],
                                            'epochs'            : epochs,
                                            'prune_type'        : prune_type_list[pt_idx],
                                            'final_nnz_weights' : final_nnz_weights_list[fnw_idx],
                                            'out_dim'           : out_dim,
                                            'in_dim'            : in_dim,
                                            'width'             : width_list[w_idx]
                                        }
                                    )