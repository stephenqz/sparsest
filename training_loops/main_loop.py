import torch
import torch.nn as nn

from cubist_spiral.CubistSpiral import get_data
from combinatorial_search.combinatorial_search import get_sub_mask, get_search_mask
from training_loops.training_utils import compute_acc, DumpJSON
from Model.FourLayerMLP import FourLayerModel
import math

from pruning_methods.Mask import Mask
from pruning_methods.Magnitude import magnitude_prune
from pruning_methods.SNIP import SNIP
from pruning_methods.GraSP import GraSP
from pruning_methods.FORCE import iterative_pruning
from pruning_methods.SynFlow import synFlow
from pruning_methods.ProsPr.prospr import ProsPrPrune
from pruning_methods.ProsPr.utils import pruning_filter_factory
from pruning_methods.rigL.rigL import RigLScheduler

from training_loops.training_utils import save_initialization


class main_loop():
    def __init__(self, exper_config, csv_path, id_path, run_id):

        for key, value in exper_config.items():
            setattr(self, key, value)
        
        self.run_id = run_id
        # fill in where you want to save results
        self.results_path = csv_path+ '/results'
        self.final_model_path = csv_path + '/checkpoints/' + str(run_id) + '.pt'
        self.id_path = id_path


    def run(self):
        model = FourLayerModel(self.in_dim, self.width, self.width, self.width, self.out_dim)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        curr_mask = Mask.ones_like(model)
        model.update_mask(curr_mask)

        iterations_per_epoch = math.floor(50000/self.batch_size)
        total_iters = self.epochs * iterations_per_epoch
        PRUNINGTIMES = 199.0
        IDXMOD = int(total_iters/(PRUNINGTIMES+1))
        
        X_train, y_train = get_data()
        train_ds = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=self.batch_size, drop_last=True, shuffle=True)
        test_loader = torch.utils.data.DataLoader(train_ds, batch_size=1024, shuffle=False)
        
        criterion = nn.BCEWithLogitsLoss()

        # setup pruning
        if self.prune_type == "comb_search":
            w1_sub_mask = get_sub_mask(self.in_dim, self.d1,  self.w1_nnz,  self.w1_mask)
            w2_sub_mask = get_sub_mask(self.d1, self.d2,  self.w2_nnz,  self.w2_mask)
            w3_sub_mask = get_sub_mask(self.d2, self.d3,  self.w3_nnz,  self.w3_mask)
            clf_sub_mask =  get_sub_mask(self.d3, self.out_dim,  self.clf_nnz,  self.clf_mask)
            curr_mask = get_search_mask(model, [self.in_dim, self.d1, self.d2, self.d3, self.out_dim], \
                                        [w1_sub_mask, w2_sub_mask, w3_sub_mask, clf_sub_mask])
            model.update_mask(curr_mask)
        
        if self.prune_type =="synFlow":
            input_dim = [2]
            curr_mask = synFlow(model, self.final_nnz_weights, input_dim)
            model.update_mask(curr_mask)
        
        elif self.prune_type == "snip":
            keep_ratio = self.final_nnz_weights/float(model.compute_weight_nnz())
            for layer in model.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_normal_(layer.weight)
            
            keep_masks = SNIP(model, keep_ratio, train_loader, criterion, device, reinit=False)
            with torch.no_grad():
                curr_mask = {}
                for l_idx in range(len(model.weight_params)):
                    weight_key = "weight_param_" + str(l_idx)
                    bias_key = "bias_param_" + str(l_idx)
                    curr_mask[weight_key] = keep_masks[l_idx]
                    if l_idx < len(model.weight_params)-1:
                            curr_mask[bias_key] = torch.sign(torch.count_nonzero(keep_masks[l_idx+1], dim=0))
                    else:
                        curr_mask[bias_key] = torch.abs(torch.sign(model.bias_params[l_idx]))
            model.update_mask(curr_mask)

        elif self.prune_type == "grasp":
            sparsity_ratio = 1.0-self.final_nnz_weights/float(model.compute_weight_nnz())
            for layer in model.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_normal_(layer.weight)
            keep_masks = GraSP(model, sparsity_ratio, train_loader, criterion, device, max(2, self.out_dim), reinit=False)
            with torch.no_grad():
                for m in model.modules():
                    if isinstance(m,nn.Linear):
                        curr_mask = keep_masks[m]
                        m.weight.data.mul_(curr_mask)
                curr_mask = {}
                for l_idx, weight in enumerate(model.weight_params):
                    weight_key = "weight_param_" + str(l_idx)
                    bias_key = "bias_param_" + str(l_idx)
                    curr_mask[weight_key] = torch.abs(torch.sign(weight))
                    if l_idx < len(model.weight_params)-1:
                            curr_mask[bias_key] = torch.sign(torch.count_nonzero(model.weight_params[l_idx+1], dim=0))
                    else:
                        curr_mask[bias_key] = torch.abs(torch.sign(model.bias_params[l_idx]))
            model.update_mask(curr_mask)
        
        elif self.prune_type =="prospr":
            prune_ratio = 1.0 - (self.final_nnz_weights/float(model.compute_weight_nnz()))
            filter_fn = pruning_filter_factory(self.out_dim, False)
            model, masks = ProsPrPrune(model, prune_ratio, train_loader, criterion, filter_fn, 3, self.lr, self.momentum)
        
        elif self.prune_type == "iter_snip":
            prune_factor = self.final_nnz_weights/float(model.compute_weight_nnz())
            keep_masks = iterative_pruning(model, train_loader, criterion, device, prune_factor, 1)
            with torch.no_grad():
                counter = 0 
                for m in model.modules():
                    if isinstance(m,nn.Linear):
                        curr_mask = keep_masks[counter]
                        m.weight.data.mul_(curr_mask)
                        counter += 1
                curr_mask = {}
                for l_idx, weight in enumerate(model.weight_params):
                    weight_key = "weight_param_" + str(l_idx)
                    bias_key = "bias_param_" + str(l_idx)
                    curr_mask[weight_key] = torch.abs(torch.sign(weight))
                    if l_idx < len(model.weight_params)-1:
                            curr_mask[bias_key] = torch.sign(torch.count_nonzero(model.weight_params[l_idx+1], dim=0))
                    else:
                        curr_mask[bias_key] = torch.abs(torch.sign(model.bias_params[l_idx]))
            model.update_mask(curr_mask)

        elif self.prune_type == "FORCE":
            prune_factor = self.final_nnz_weights/float(model.compute_weight_nnz())
            keep_masks = iterative_pruning(model, train_loader, criterion, device, prune_factor, 3)
            with torch.no_grad():
                counter = 0 
                for m in model.modules():
                    if isinstance(m,nn.Linear):
                        curr_mask = keep_masks[counter]
                        m.weight.data.mul_(curr_mask)
                        counter += 1
                curr_mask = {}
                for l_idx, weight in enumerate(model.weight_params):
                    weight_key = "weight_param_" + str(l_idx)
                    bias_key = "bias_param_" + str(l_idx)
                    curr_mask[weight_key] = torch.abs(torch.sign(weight))
                    if l_idx < len(model.weight_params)-1:
                            curr_mask[bias_key] = torch.sign(torch.count_nonzero(model.weight_params[l_idx+1], dim=0))
                    else:
                        curr_mask[bias_key] = torch.abs(torch.sign(model.bias_params[l_idx]))
            model.update_mask(curr_mask)
        
        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)

        if self.lr_milestones != "cosine_annealing":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, self.lr_milestones, gamma=0.1)
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epochs)
        
        if self.prune_type == "rigL":
            rigL_pruner_dict = None
            T_end = int(0.75 * total_iters)
            dense_allocation = self.final_nnz_weights/model.compute_weight_nnz()
            pruner = RigLScheduler(model,                          
                    optimizer,                         
                    dense_allocation=dense_allocation, 
                                                        
                    sparsity_distribution='ERK',     
                    T_end=T_end,                    
                    delta=100,                      
                    alpha=0.3,                       
                    grad_accumulation_n=1,          
                                                       
                    static_topo=False,               
                                                       
                    state_dict=rigL_pruner_dict)
    
        curr_iteration = 0
        num_of_prunable = model.compute_weight_nnz()
        model.train()
        for epoch in range(1, self.epochs+1):
            for iter_idx, (data, target) in enumerate(train_loader):
                if curr_iteration % IDXMOD == 0:
                    if curr_iteration > 0:
                        # Do IMP Pruning
                        if self.prune_type == "imp":
                            model.apply_mask()
                            sparsity_frac = (curr_iteration)/(PRUNINGTIMES*IDXMOD)
                            curr_params_remain = self.final_nnz_weights + (num_of_prunable - self.final_nnz_weights)*((1-sparsity_frac)**3)
                            curr_mask = magnitude_prune(model, int(curr_params_remain))
                            model.update_mask(curr_mask)
                        model.train()

                data, target = data.to(device), target.to(device)
                model.apply_mask()

                optimizer.zero_grad()
                logits = model(data)
                loss = criterion(logits, target)
                loss.backward()
                if self.prune_type == "rigL":
                    if pruner():
                        optimizer.step()
                else:
                    optimizer.step()
                
                model.apply_mask()
                curr_iteration += 1
            scheduler.step()

        with torch.no_grad():
            acc = compute_acc(test_loader, model).item()
        
        final_stats = {
        'Model NNZ': model.compute_nnz(),
        'Weight NNZ': model.compute_weight_nnz(),
        'Accuracy': acc,
        }

        results = DumpJSON(read_path=(self.results_path+'.json'),
                        write_path=(self.results_path+'.json'))
        results.append(dict(self.__getstate__(), **final_stats))
        results.save()
        results.to_csv()

        # save final model
        save_initialization(model, self.final_model_path) 

    def __getstate__(self):
        state = self.__dict__.copy()

        # remove fields that should not be saved
        attributes = ['results_path',
                      'id_path',
        'device',
        'final_model_path'
        ]

        for attr in attributes:
            try:
                del state[attr]
            except:
                pass

        return state

