import torch
import torch.nn as nn

class FourLayerModel(nn.Module):
    def __init__(self, d_in, d_1, d_2, d_3, num_classes):
        super().__init__()

        self.first_layer = nn.Linear(d_in, d_1, bias=True)
        self.second_layer = nn.Linear(d_1, d_2, bias=True)
        self.third_layer = nn.Linear(d_2, d_3, bias=True)
        self.clf = nn.Linear(d_3, num_classes, bias=True)

        self.weight_params = [self.first_layer.weight, self.second_layer.weight, self.third_layer.weight, self.clf.weight]
        self.bias_params = [self.first_layer.bias, self.second_layer.bias, self.third_layer.bias, self.clf.bias]

        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.first_layer(x)
        x = self.relu(x)
        x = self.second_layer(x)
        x = self.relu(x)
        x = self.third_layer(x)
        x = self.relu(x)
        x = self.clf(x)
        return x

    def compute_nnz(self):
        self.apply_mask()
        return sum([torch.count_nonzero(param).item() for param in self.parameters() if param is not None])
    
    def compute_weight_nnz(self):
        self.apply_mask()
        return sum([torch.count_nonzero(param).item() for param in self.weight_params])
    
    #=================================================================================================#
    # Code based on: https://github.com/facebookresearch/open_lth
     
    # Masking Functions
    def update_mask(self, mask):
        for k, v in mask.items(): self.register_buffer(to_mask_name(k), v.float())
        self.apply_mask()

    def apply_mask(self):
        with torch.no_grad():
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            for weight_idx, weight in enumerate(self.weight_params):
                key = "weight_param_" + str(weight_idx)
                if hasattr(self, to_mask_name(key)):
                    curr_mask = getattr(self, to_mask_name(key)).to(device)
                    weight.data *= curr_mask
            
            for bias_idx, bias in enumerate(self.bias_params):
                key = "bias_param_" + str(bias_idx)
                if hasattr(self, to_mask_name(key)):
                    curr_mask = getattr(self, to_mask_name(key)).to(device)
                    bias.data *= curr_mask
    
    def print_mask(self):
        with torch.no_grad():
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            for weight_idx, weight in enumerate(self.weight_params):
                key = "weight_param_" + str(weight_idx)
                if hasattr(self, to_mask_name(key)):
                    curr_mask = getattr(self, to_mask_name(key)).to(device)
                    print(curr_mask)
            
            for bias_idx, bias in enumerate(self.bias_params):
                key = "bias_param_" + str(bias_idx)
                if hasattr(self, to_mask_name(key)):
                    curr_mask = getattr(self, to_mask_name(key)).to(device)
                    print(curr_mask)

def to_mask_name(name):
    return 'mask_' + name.replace('.', '___')
    #=================================================================================================#