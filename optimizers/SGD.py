from __future__ import annotations
import torch
from optimizers.optim import Optimizer

class SGD(Optimizer):
    def __init__(self, params, lr: float = 0.1):
        super().__init__(params)
        self.lr = lr
    
    def step(self):
        for p in self.params:
            if p.grad is None:
                continue

            p.data -= self.lr * p.grad
