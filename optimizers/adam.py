from __future__ import annotations
import torch
from optimizers.optim import Optimizer

class AdaGrad(Optimizer):
    def __init__(self, params,lr=0.01,eps=1e-8):
        super().__init__(params,lr)
        self.eps = eps

        self.state = {
            id(p): {"G" : torch.zeros_like(p.data)}
            for group in self.params_group
            for p in group["params"]
            }
    
    def step(self):
        for group in self.params_group:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[id(p)]
                G = state["G"]

                G.addcmul_(p.grad,p.grad,value=1)
                p.data.add_(-lr * p.grad / (G.sqrt() + self.eps))

class RMSProp(Optimizer):
    def __init__(self, params, lr= 0.01, alpha = 0.9,eps = 1e-8):
        super().__init__(params, lr)
        self.eps = eps
        self.alpha = alpha

        self.state = {id(p): {"E": torch.zeros_like(p.data)}
                      for group in self.params_group
                      for p in group["params"]}
        
    def step(self):
        for group in self.params_group:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[id(p)]
                E = state["E"]

                E.mul_(self.alpha).addcmul_(p.grad,p.grad,value=(1 - self.alpha))
                p.data.add_(-lr * p.grad / E.sqrt() + self.eps)