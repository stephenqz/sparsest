import sys

from experiment_configurations.pruning_experiments.fixed_init_configs.prune_95_configs import prune_95_exper
from experiment_configurations.comb_search_experiments.fixed_init_configs.first_phase_95_configs import first_phase_95_experiments
from experiment_configurations.comb_search_experiments.fixed_init_configs.second_phase_95_configs import second_phase_95_exper

from training_loops.fixed_init_exper.main_loop import main_loop_fixed_init
from training_loops.fixed_init_exper.LTH_loop import LTH_loop_fixed_init


seed = int(sys.argv[1])

# PLEASE FILL IN
csv_path = "./csv_path"
id_path = "./id_path"

init_path_95 = './BENCH-95/initialization_checkpoint.pt'

fixed_init_95_exper = prune_95_exper + first_phase_95_experiments + second_phase_95_exper

if fixed_init_95_exper[seed-1]['prune_type'] == "imp_LTH":
    exp = LTH_loop_fixed_init(fixed_init_95_exper[seed-1], csv_path, id_path, init_path_95, seed)
else:
    exp = main_loop_fixed_init(fixed_init_95_exper[seed-1], csv_path, id_path, init_path_95, seed)
exp.run()