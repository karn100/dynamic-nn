import torch
from __future__ import annotations
from core_mlp.module import Module  

class ReLU(Module):
    # ReLU = max(0,x) -- if x>0 then x else 0
    def forward(slef,x:torch.Tensor) -> torch.Tensor:
        return torch.clamp_min(x, 0.0)   #can alose write as torch.maximum(x,torch.tensor(0.0))

class Sigmoid(Module):
    # sigmoid = 1 / (1 + exp(-x))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x)

class Tanh(Module):
    #Tanh = tanh(x)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x)