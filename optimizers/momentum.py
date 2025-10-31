from __future__ import annotations
import torch
from optimizers.optim import Optimizer

class Momentum(Optimizer):
    def __init__(self, params, lr: float = 0.1, momentum: float = 0.9):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum

        self.velocities = {p: torch.zeros_like(p) for p in self.params}
    
    def step(self):
        for p in self.params:
            if p.grad is None:
                continue
            v = self.velocities[p]
            grad = p.grad

            v.mul_(self.momentum).add_(grad,alpha=(1 - self.momentum))
            p.data.add_(v,alpha=-self.lr)


class MomentumwithNAG(Optimizer):
    def __init__(self, params,lr: float = 0.1,momentum: float = 0.9):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum

        # set initial velcities of paramteres(weights,bias) to 0 with same size as theirs in dict.
        self.velocities = {p: torch.zeros_like(p) for p in self.params}

    def step(self):
        for p in self.params:
            if p.grad is None:
                continue

            v = self.velocities[p]
            grad = p.grad

            # we need to lookahead using previous velocity as it is the v we already know and we use 
            # that to compute a lookahead gradient to make sure what will be the direction of our gradient.
            v_prev = v.clone()

            # v calculation 
            v.mul_(self.momentum).add_(grad)

            # here we use practical approach to update parameter using lookahead 
            # theta = theta - lr*(grad(theta) + beta*v_prev) --> (theta - lr*grad(theta)) = lookahead
            # theta = lookahead - lr*beta*v_prev
            p.data.add_(v_prev,alpha=-self.momentum*self.lr).add_(grad,alpha=-self.lr)

            self.velocities[p] = v
        