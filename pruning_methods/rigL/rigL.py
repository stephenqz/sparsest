""" 
Code based on: https://github.com/verbocado/rigl-torch
"""

import numpy as np
import torch
import torch.distributed as dist

from pruning_methods.rigL.util import get_W, get_B
from pruning_methods.rigL.ERK import ERK


class IndexMaskHook:
    def __init__(self, layer, scheduler):
        self.layer = layer
        self.scheduler = scheduler
        self.dense_grad = None

    def __name__(self):
        return 'IndexMaskHook'

    @torch.no_grad()
    def __call__(self, grad):
        mask = self.scheduler.backward_masks[self.layer]
        if self.scheduler.check_if_backward_hook_should_accumulate_grad():
            if self.dense_grad is None:
                self.dense_grad = torch.zeros_like(grad)
            self.dense_grad += grad / self.scheduler.grad_accumulation_n
        else:
            self.dense_grad = None

        return grad * mask


def _create_step_wrapper(scheduler, optimizer):
    _unwrapped_step = optimizer.step
    def _wrapped_step():
        _unwrapped_step()
        scheduler.reset_momentum()
        scheduler.apply_mask_to_weights()
    optimizer.step = _wrapped_step


class RigLScheduler:

    def __init__(self, model, optimizer, dense_allocation=1, T_end=None, sparsity_distribution='uniform', delta=100, alpha=0.3, static_topo=False, grad_accumulation_n=1, state_dict=None):
        if dense_allocation <= 0 or dense_allocation > 1:
            raise Exception('Dense allocation must be on the interval (0, 1]. Got: %f' % dense_allocation)

        self.model = model
        self.optimizer = optimizer

        self.W = get_W(model)
        self.biases = get_B(model)

        _create_step_wrapper(self, optimizer)
            
        self.dense_allocation = dense_allocation
        self.N = [torch.numel(w) for w in self.W]

        if state_dict is not None:
            self.load_state_dict(state_dict)
            self.apply_mask_to_weights()

        else:
            self.sparsity_distribution = sparsity_distribution
            self.static_topo = static_topo
            self.grad_accumulation_n = grad_accumulation_n
            self.backward_masks = None
            self.bias_masks = None

            if sparsity_distribution == 'ERK':
                dense_allocations = ERK(self.W, density=dense_allocation)

            # define sparsity allocation
            self.S = []
            for i, W in enumerate(self.W):
                is_first_layer = i == 0
                if is_first_layer and self.sparsity_distribution == 'uniform' and len(self.W) > 1:
                    self.S.append(0)
                else:
                    if self.sparsity_distribution == 'uniform':
                        self.S.append(1-dense_allocation)
                    elif self.sparsity_distribution == 'ERK':
                        self.S.append(1 - dense_allocations[i])

            self.random_sparsify()

            self.step = 0
            self.rigl_steps = 0

            self.delta_T = delta
            self.alpha = alpha
            self.T_end = T_end

        self.backward_hook_objects = []
        for i, w in enumerate(self.W):
            if self.S[i] <= 0:
                self.backward_hook_objects.append(None)
                continue

            if getattr(w, '_has_rigl_backward_hook', False):
                raise Exception('This model already has been registered to a RigLScheduler.')
        
            self.backward_hook_objects.append(IndexMaskHook(i, self))
            w.register_hook(self.backward_hook_objects[-1])
            setattr(w, '_has_rigl_backward_hook', True)

        assert self.grad_accumulation_n > 0 and self.grad_accumulation_n < delta
        assert self.sparsity_distribution in ('uniform', 'ERK',)




    def state_dict(self):
        obj = {
            'dense_allocation': self.dense_allocation,
            'S': self.S,
            'N': self.N,
            'hyperparams': {
                'delta_T': self.delta_T,
                'alpha': self.alpha,
                'T_end': self.T_end,
                'static_topo': self.static_topo,
                'sparsity_distribution': self.sparsity_distribution,
                'grad_accumulation_n': self.grad_accumulation_n,
            },
            'step': self.step,
            'rigl_steps': self.rigl_steps,
            'backward_masks': self.backward_masks,
        }

        return obj

    def load_state_dict(self, state_dict):
        for k, v in state_dict.items():
            if type(v) == dict:
                self.load_state_dict(v)
            setattr(self, k, v)


    @torch.no_grad()
    def random_sparsify(self):
        is_dist = dist.is_initialized()
        self.backward_masks = []
        self.bias_masks = []
        for l, w in enumerate(self.W):
            if self.S[l] <= 0:
                self.backward_masks.append(None)
                continue

            n = self.N[l]
            s = int(self.S[l] * n)
            perm = torch.randperm(n)
            perm = perm[:s]
            flat_mask = torch.ones(n, device=w.device)
            flat_mask[perm] = 0
            mask = torch.reshape(flat_mask, w.shape)

            if is_dist:
                dist.broadcast(mask, 0)

            mask = mask.bool()
            w *= mask
            self.backward_masks.append(mask)
        
        for bias_idx, b in enumerate(self.biases):
            if bias_idx < len(self.backward_masks)-1:
                mask = self.backward_masks[bias_idx + 1]
                if mask is not None:
                    self.bias_masks.append(torch.sign(torch.count_nonzero(mask, dim=0)))
                else:
                    self.bias_masks.append(torch.ones_like(b, device=b.device))
            else:
                self.bias_masks.append(torch.ones_like(b, device=b.device))

    def __str__(self):
        s = 'RigLScheduler(\n'
        s += 'layers=%i,\n' % len(self.N)

        N_str = '['
        S_str = '['
        sparsity_percentages = []
        total_params = 0
        total_nonzero = 0

        for N, S, mask, W in zip(self.N, self.S, self.backward_masks, self.W):
            actual_S = torch.sum(W[mask == 0] == 0).item()
            N_str += ('%i/%i, ' % (N-actual_S, N))
            sp_p = float(N-actual_S) / float(N) * 100
            S_str += '%.2f%%, ' % sp_p
            sparsity_percentages.append(sp_p)
            total_params += N
            total_nonzero += N-actual_S

        N_str = N_str[:-2] + ']'
        S_str = S_str[:-2] + ']'
        
        s += 'nonzero_params=' + N_str + ',\n'
        s += 'nonzero_percentages=' + S_str + ',\n'
        s += 'total_nonzero_params=' + ('%i/%i (%.2f%%)' % (total_nonzero, total_params, float(total_nonzero)/float(total_params)*100)) + ',\n'
        s += 'step=' + str(self.step) + ',\n'
        s += 'num_rigl_steps=' + str(self.rigl_steps) + ',\n'
        s += 'sparsity_distribution=' + str(self.sparsity_distribution) + ',\n'

        return s + ')'


    @torch.no_grad()
    def reset_momentum(self):
        for w, mask, s in zip(self.W, self.backward_masks, self.S):
            if s <= 0:
                continue

            param_state = self.optimizer.state[w]
            if 'momentum_buffer' in param_state:
                buf = param_state['momentum_buffer']
                buf *= mask
        for b, mask in zip(self.biases, self.bias_masks):
            param_state = self.optimizer.state[b]
            if 'momentum_buffer' in param_state:
                buf = param_state['momentum_buffer']
                buf *= mask


    @torch.no_grad()
    def apply_mask_to_weights(self):
        for w, mask, s in zip(self.W, self.backward_masks, self.S):
            if s <= 0:
                continue
            w *= mask
        
        for b, mask in zip(self.biases, self.bias_masks):
            b *= mask

    @torch.no_grad()
    def apply_mask_to_gradients(self):
        for w, mask, s in zip(self.W, self.backward_masks, self.S):
            if s <= 0:
                continue
            w.grad *= mask
        
        for b, mask in zip(self.biases, self.bias_masks):
            b.grad *= mask

    
    def check_if_backward_hook_should_accumulate_grad(self):
        """
        Used by the backward hooks. Basically just checks how far away the next rigl step is, 
        if it's within `self.grad_accumulation_n` steps, return True.
        """

        if self.step >= self.T_end:
            return False

        steps_til_next_rigl_step = self.delta_T - (self.step % self.delta_T)
        return steps_til_next_rigl_step <= self.grad_accumulation_n


    def cosine_annealing(self):
        return self.alpha / 2 * (1 + np.cos((self.step * np.pi) / self.T_end))


    def __call__(self):
        self.step += 1
        if self.static_topo:
            return True
        if (self.step % self.delta_T) == 0 and self.step < self.T_end:
            self._rigl_step()
            self.rigl_steps += 1
            return False
        return True


    @torch.no_grad()
    def _rigl_step(self):
        drop_fraction = self.cosine_annealing()

        is_dist = dist.is_initialized()
        world_size = dist.get_world_size() if is_dist else None

        for l, w in enumerate(self.W):
            if self.S[l] <= 0:
                continue

            current_mask = self.backward_masks[l]

            score_drop = torch.abs(w)
            score_grow = torch.abs(self.backward_hook_objects[l].dense_grad)

            if is_dist:
                dist.all_reduce(score_drop)  
                score_drop /= world_size     

                dist.all_reduce(score_grow)  
                score_grow /= world_size     

            n_total = self.N[l]
            n_ones = torch.sum(current_mask).item()
            n_prune = int(n_ones * drop_fraction)
            n_keep = n_ones - n_prune

            _, sorted_indices = torch.topk(score_drop.view(-1), k=n_total)
            new_values = torch.where(
                            torch.arange(n_total, device=w.device) < n_keep,
                            torch.ones_like(sorted_indices),
                            torch.zeros_like(sorted_indices))
            mask1 = new_values.scatter(0, sorted_indices, new_values)

            score_grow = score_grow.view(-1)

            score_grow_lifted = torch.where(
                                mask1 == 1, 
                                torch.ones_like(mask1) * (torch.min(score_grow) - 1),
                                score_grow)

            _, sorted_indices = torch.topk(score_grow_lifted, k=n_total)
            new_values = torch.where(
                            torch.arange(n_total, device=w.device) < n_prune,
                            torch.ones_like(sorted_indices),
                            torch.zeros_like(sorted_indices))
            mask2 = new_values.scatter(0, sorted_indices, new_values)

            mask2_reshaped = torch.reshape(mask2, current_mask.shape)
            grow_tensor = torch.zeros_like(w)
            
            REINIT_WHEN_SAME = False
            if REINIT_WHEN_SAME:
                raise NotImplementedError()
            else:
                new_connections = ((mask2_reshaped == 1) & (current_mask == 0))

            new_weights = torch.where(new_connections.to(w.device), grow_tensor, w)
            w.data = new_weights

            mask_combined = torch.reshape(mask1 + mask2, current_mask.shape).bool()

            current_mask.data = mask_combined

        #mask biases
        for bias_idx, b in enumerate(self.biases):
            if bias_idx < len(self.backward_masks)-1:
                mask = self.backward_masks[bias_idx + 1]
                if mask is not None:
                    self.bias_masks[bias_idx] = torch.sign(torch.count_nonzero(mask, dim=0))
                else:
                    self.bias_masks[bias_idx] = torch.ones_like(b, device=b.device)
            else:
                self.bias_masks[bias_idx] = torch.ones_like(b, device=b.device)

        self.reset_momentum()
        self.apply_mask_to_weights()
        self.apply_mask_to_gradients() 