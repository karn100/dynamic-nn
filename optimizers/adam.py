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

class Adam(Optimizer):
    def __init__(self, params, lr = 1e-3,beta1 = 0.9,beta2 = 0.999,eps = 1e-8):
        super().__init__(params, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.state = {id(p): {"m":torch.zeros_like(p.data),"v": torch.zeros_like(p.data),"t":0}
                      for group in self.params_group
                      for p in group["params"]}
    
    def step(self):
        for group in self.params_group:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                g = p.grad
                state = self.state[id(p)]

                v = state["v"]
                m = state["m"]

                state["t"] += 1
                t = state["t"]

                m.mul_(self.beta1).add_(g, alpha = (1 - self.beta1))
                v.mul_(self.beta2).addcmul_(g,g, value = (1 - self.beta2))

                m_hat = m / (1 - self.beta1**t)
                v_hat = v / (1 - self.beta2**t)

                p.data.add_(-lr * m_hat / (torch.sqrt(v_hat) + self.eps))
