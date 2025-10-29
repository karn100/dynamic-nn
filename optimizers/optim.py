from __future__ import annotations
import torch
from typing import Iterable

class Optimizer:
    def __init__(self,params: Iterable[torch.Tensor]):
        self.params = list(params)
    
    def zero_grad(self):
        # This is because, during backward pass- we dont overwrite grads but accumulate them, so if we have previous computed grad,
        # it will be added to the next computed grad which will make training unstable and messy,
        # So, thats why we zero the previous grad and it will not make any impact on the current grad

        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()
    def step(self):

        #It performs single optimization step
        # This is the function that we overwriite according to which optimizer we are implementing

        raise NotImplementedError