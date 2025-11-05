from __future__ import annotations
import torch
from optimizers.optim import Optimizer

class SGD(Optimizer):
    def __init__(self, params, lr: float = 0.1):
        super().__init__(params,lr)
        
    
    def step(self):
        for group in self.params_group:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data -= lr * p.grad
