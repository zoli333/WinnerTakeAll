import torch
import torch.nn as nn
import math
from typing import List, Optional
from torch import Tensor

# https://www.geeksforgeeks.org/custom-optimizers-in-pytorch/
# https://github.com/pytorch/pytorch/blob/aab67c6dff3819d3c05b328e964f1642227adbe6/torch/optim/_functional.py#L156


class MomentumOptimizer(torch.optim.Optimizer):

    # Init Method: 
    def __init__(self, params, lr=1e-3, momentum=0.9, weight_decay=0.0):
        super(MomentumOptimizer, self).__init__(params, defaults={'lr': lr})
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.state = dict()
        for group in self.param_groups:
            for p in group['params']:
                self.state[p] = dict(mom=torch.zeros_like(p.data))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if p not in self.state:
                    self.state[p] = dict(mom=torch.zeros_like(p.data))

                mom = self.state[p]['mom']

                mom.mul_(self.momentum)
                grad.mul_(-group['lr'])
                mom.add_(grad)

                p.add_(mom)

                if self.weight_decay > 0.0:
                    p.add_(p, alpha=(-group["lr"] * self.weight_decay))

