import math

prune_995_exper = []

lr_list = [0.05, 0.1, 0.2]
weight_decay_list = [5e-4]
momentum_list = [0.9]
batch_size_list = [128]

width_list = [6, 16]
prune_type_list = ["iter_snip", "prospr", "FORCE", "imp", "imp_LTH", "grasp", "snip", "rigL", "synFlow"]
lr_milestones_list = ["cosine_annealing", [0.3, 0.6], []]
final_nnz_weights_list = [16, 18, 20, 22, 24, 25, 26, 27, 28, 29, 30, 32, 34, 36]
out_dim = 1
in_dim = 2

num_runs = 5
counter = 0
while counter < num_runs:
    for lr_idx in range(len(lr_list)):
        for bs_idx in range(len(batch_size_list)):
            for lrML_idx in range(len(lr_milestones_list)):
                for mom_idx in range(len(momentum_list)):
                    for pt_idx in range(len(prune_type_list)):
                        for wd_idx in range(len(weight_decay_list)):
                            epoch_percents = lr_milestones_list[lrML_idx]
                            if prune_type_list[pt_idx] == "imp_LTH":
                                epochs = 250
                                if epoch_percents != "cosine_annealing":
                                    lr_milestones = [math.ceil(50*curr_percent) for curr_percent in epoch_percents]
                                else:
                                    lr_milestones = "cosine_annealing"
                            else:
                                epochs = 50
                                if epoch_percents != "cosine_annealing":
                                    lr_milestones = [math.ceil(epochs*curr_percent) for curr_percent in epoch_percents]
                                else:
                                    lr_milestones = "cosine_annealing"
                            for w_idx in range(len(width_list)):
                                for fnw_idx in range(len(final_nnz_weights_list)):
                                    prune_995_exper.append(
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
    counter+=1