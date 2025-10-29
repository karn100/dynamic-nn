from __future__ import annotations
import torch 
from core_mlp.module import Module

class Dropout(Module):
    def __init__(self,p: float = 0.5):
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError("Dropout probability must be betwwn [0,1)]")
        self.p = p
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        
        mask = (torch.rand_like(x) > self.p).float()
        return mask * x / (1.0 - self.p)
    