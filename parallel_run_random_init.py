import sys

from experiment_configurations.pruning_experiments.overparam_prune_configs import overparam_prune_exper
from experiment_configurations.comb_search_experiments.first_phase_configs import first_phase_experiments
from experiment_configurations.comb_search_experiments.second_phase_95_configs import second_phase_95_exper
from experiment_configurations.comb_search_experiments.second_phase_995_configs import second_phase_995_exper

from training_loops.main_loop import main_loop
from training_loops.LTH_loop import LTH_loop


seed = int(sys.argv[1])

# PLEASE FILL IN
csv_path = "./csv_path"
id_path = "./id_path"

random_init_exper = overparam_prune_exper + first_phase_experiments + second_phase_95_exper + second_phase_995_exper

if random_init_exper[seed-1]['prune_type'] == "imp_LTH":
    exp = LTH_loop(random_init_exper[seed-1], csv_path, id_path, seed)
else:
    exp = main_loop(random_init_exper[seed-1], csv_path, id_path, seed)
exp.run()