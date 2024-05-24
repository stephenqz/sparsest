import torch
import torch.nn as nn
from cubist_spiral.CubistSpiral import get_data
from training_loops.training_utils import compute_acc, DumpJSON, save_initialization, rewind_model
from pruning_methods.Magnitude import magnitude_prune
from Model.FourLayerMLP import FourLayerModel
from pruning_methods.Mask import Mask

class LTH_loop():
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

        save_initialization(model, self.id_path + '/initialization_chkpt.pt')
        
        X_train, y_train = get_data()
        train_ds = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=self.batch_size, drop_last=True, shuffle=True)
        test_loader = torch.utils.data.DataLoader(train_ds, batch_size=1024, shuffle=False)
        
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)

        if self.lr_milestones != "cosine_annealing":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, self.lr_milestones, gamma=0.1)
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 50)
    
        num_of_prunable = model.compute_weight_nnz()

        model.train()
        for epoch in range(1, self.epochs+1):
            for iter_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                model.apply_mask()

                optimizer.zero_grad()
                logits = model(data)
                loss = criterion(logits, target)
                loss.backward()
                optimizer.step()
                
                model.apply_mask()
            scheduler.step()

            if epoch % 50 == 0:
                if epoch != self.epochs:
                    times_pruned = self.epochs//50 - 1
                    curr_prune_mile = epoch//50
                    power = (curr_prune_mile/times_pruned)
                    curr_params_remain = num_of_prunable * ((float(self.final_nnz_weights)/num_of_prunable)**power)
                    curr_mask = magnitude_prune(model, int(curr_params_remain))
                    
                    model = rewind_model(model, self.id_path + '/initialization_chkpt.pt')

                    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
                    if self.lr_milestones != "cosine_annealing":
                        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, self.lr_milestones, gamma=0.1)
                    else:
                        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 50)
                    
                    model.update_mask(curr_mask)
                    model.apply_mask()
                    model.train()

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

