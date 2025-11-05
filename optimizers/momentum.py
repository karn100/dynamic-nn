from __future__ import annotations
import torch
from optimizers.optim import Optimizer

class Momentum(Optimizer):
    def __init__(self,params,lr,momentum = 0.9):
        super().__init__(params,lr)
        self.momentum = momentum

        self.state = {id(p):{"velocity":None}
                      for group in self.params_group
                      for p in group["params"]}
    
    def step(self):
        for group in self.params_group:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[id(p)]
                v = state["velocity"]

                if v is None:
                    v = state["velocity"] = p.grad.clone()
                
                v.mul_(self.momentum).add_(p.grad)
                p.data -= lr * v 

class MomentumwithNAG(Optimizer):
    def __init__(self, params,lr,momentum = 0.9):
        super().__init__(params,lr)
        self.momentum = momentum
        # set initial velcities of paramteres(weights,bias) to 0 with same size as theirs in dict.
        
        self.state = {id(p): {"velocity" : None}
                      for group in self.params_group
                      for p in group["params"]}

    def step(self):
       for group in self.params_group:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[id(p)]
                v = state["velocity"]

                if v is None:
                    v = state["velocity"] = p.grad.clone()

                v.mul_(self.momentum).add_(p.grad)
                    
                p.data -= lr*(self.momentum * v + p.grad) 