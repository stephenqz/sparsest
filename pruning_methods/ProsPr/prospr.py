"""
Code based on: https://github.com/mil-ad/prospr
"""

import types
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import pruning_methods.ProsPr.utils as utils

__all__ = ["prune"]


def functional_sgd(
    params: List[torch.Tensor],
    grads: List[torch.Tensor],
    momentum_state: List[Optional[torch.Tensor]],
    lr: float,
    momentum: float = 0,
    weight_decay: float = 0,
    dampening: float = 0,
    nesterov: bool = False,
) -> Tuple[List[torch.Tensor], List[Optional[torch.Tensor]]]:

    updated_params = []
    updated_momentum_state = []

    for i, param in enumerate(params):
        grad = grads[i]

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        if momentum != 0:
            buf = momentum_state[i]

            if buf is None:
                buf = torch.clone(grad).detach()
            else:
                buf = buf.mul(momentum).add(grad, alpha=1 - dampening)
            updated_momentum_state.append(buf)

            if nesterov:
                grad = grad.add(buf, alpha=momentum)
            else:
                grad = buf

        updated_params.append(param.add(grad, alpha=-lr))

    return updated_params, updated_momentum_state


def fwd_with_external_params(
    net: nn.Module, x: torch.Tensor, params: List[torch.Tensor]
):
    """
    Allows calling networks's forward with parameters as arguments instead of relying on
    parameters stored in the network as state. Works by monkey-patching the forward
    methods of Linear, Conv2D, and BatchNorm2D layers such that they use the params from
    the provided argument instead of module attrbiutes. Will complain if it encounters
    any other type of modules.
    """

    def forward_factory(layer, weight, bias):
        def conv2d_fwd_external_params(self, x):
            return F.conv2d(
                x,
                weight,
                bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )

        def linear_fwd_external_params(self, x):
            return F.linear(x, weight, bias)

        def batchnorm2d_fwd_external_params(self, x):
            self._check_input_dim(x)
            if self.momentum is None:
                exponential_average_factor = 0.0
            else:
                exponential_average_factor = self.momentum
            if self.training and self.track_running_stats:
                if self.num_batches_tracked is not None:
                    self.num_batches_tracked = self.num_batches_tracked + 1
                    if self.momentum is None:
                        exponential_average_factor = 1.0 / float(
                            self.num_batches_tracked
                        )
                    else:
                        exponential_average_factor = self.momentum
            if self.training:
                bn_training = True
            else:
                bn_training = (self.running_mean is None) and (self.running_var is None)

            return F.batch_norm(
                x,
                self.running_mean
                if not self.training or self.track_running_stats
                else None,
                self.running_var
                if not self.training or self.track_running_stats
                else None,
                weight,
                bias,
                bn_training,
                exponential_average_factor,
                self.eps,
            )

        if isinstance(layer, nn.Conv2d):
            return conv2d_fwd_external_params
        elif isinstance(layer, nn.Linear):
            return linear_fwd_external_params
        elif isinstance(layer, nn.BatchNorm2d):
            return batchnorm2d_fwd_external_params
        else:
            raise TypeError

    modules_params = defaultdict(lambda: {})
    param_names = [n for (n, _) in net.named_parameters() if "weight_mask" not in n]
    for n, p in zip(param_names, params):
        module = utils.get_module_from_param_name(net, n)
        if "weight" in n:
            modules_params[module]["weight"] = p
        elif "bias" in n:
            modules_params[module]["bias"] = p
        else:
            assert False

        modules_params[module]["old_fwd"] = module.forward

    for module, module_params in modules_params.items():
        weight = module_params["weight"]
        bias = module_params["bias"] if "bias" in module_params else None
        module.forward = types.MethodType(forward_factory(module, weight, bias), module)

    out = net(x)

    for module, module_params in modules_params.items():
        module.forward = module_params["old_fwd"]

    return out


def compute_meta_grads(net, criterion, x, y, fast_params, detailed_params, weight_masks, method):
    y_pred = fwd_with_external_params(net, x, fast_params)
    loss = criterion(y_pred, y)

    if method == "full":
        meta_grads = torch.autograd.grad(
            loss, weight_masks, create_graph=False, retain_graph=True
        )
    elif method == "first_order":
        grads_fo = torch.autograd.grad(loss, fast_params, create_graph=False)
        meta_grads = []
        for g, fp, (param_name, module, _) in zip(
            grads_fo, fast_params, detailed_params
        ):
            if param_name.endswith(".weight") and hasattr(module, "weight_mask"):
                meta_grads.append(
                    torch.autograd.grad(fp, module.weight_mask, g, create_graph=False)[
                        0
                    ]
                )

    return meta_grads


def ProsPrPrune(
    net: nn.Module,
    prune_ratio,
    dataloader,
    criterion,
    filter_fn: Callable,
    num_steps: int,
    inner_lr: float,
    inner_momentum: float,
    method: str = "full",
    structured: bool = False,
    new_data_in_inner: bool = True,
    return_scores: bool = False,
) -> List[torch.Tensor]:

    assert method in ("full", "first_order")

    start_time = datetime.now()
    device_orig = next(net.parameters()).device
    net_orig = net

    net = deepcopy(net)
    net.train()
    if method == "full":
        net = net.cpu()

    device = next(net.parameters()).device
    detailed_params = [
        (n, utils.get_module_from_param_name(net, n), p)
        for (n, p) in net.named_parameters()
    ]

    weight_masks = utils.attach_masks_as_parameter(
        net,
        filter_fn,
        structured=structured,
        gradient_tie=False,
        make_weights_constants=False,
        override_forward=False,
    )

    scores = [torch.zeros_like(mask) for mask in weight_masks]

    fast_params = []
    for param_name, module, param in detailed_params:
        if param_name.endswith(".weight") and hasattr(module, "weight_mask"):
            fast_params.append(param * module.weight_mask)
        else:
            fast_params.append(param.clone())

    momentum_state = [None for _ in fast_params]

    outer_dataloader = iter(dataloader)
    x, y = next(outer_dataloader)
    x, y = x.to(device), y.to(device)

    for i in range(num_steps):

        y_pred = fwd_with_external_params(net, x, fast_params)
        inner_loss = criterion(y_pred, y)
        inner_grads = torch.autograd.grad(
            inner_loss,
            fast_params,
            create_graph=False if method == "first_order" else True,
        )

        print(f"ProsPr step {i:2} - loss {inner_loss:.4f}")

        fast_params, momentum_state = functional_sgd(
            fast_params,
            inner_grads,
            momentum_state,
            lr=inner_lr,
            momentum=inner_momentum,
        )

        if new_data_in_inner:
            try:
                x, y = next(outer_dataloader)
            except StopIteration:
                outer_dataloader = iter(dataloader)
                x, y = next(outer_dataloader)

            x, y = x.to(device), y.to(device)

    meta_grads = compute_meta_grads(
        net, criterion, x, y, fast_params, detailed_params, weight_masks, method
    )

    scores = [curr + g.abs() for curr, g in zip(scores, meta_grads)]

    end_time = datetime.now()
    print(f"⏰ Finished ProsPr in {end_time - start_time}")

    if return_scores:
        return [score.to(device_orig) for score in scores]

    scores_vec = torch.cat([score.flatten() for score in scores])
    keep_masks = utils.keep_mask_from_scores(scores_vec, 1 - prune_ratio, weight_masks)

    keep_masks = [mask.to(device_orig) for mask in keep_masks]

    utils.keep_masks_stats(keep_masks)
    utils.keep_masks_health_check(keep_masks)

    pruned_net = utils.apply_masks_with_hooks(
        net_orig, keep_masks, structured, filter_fn
    )

    return pruned_net, keep_masks