from __future__ import annotations
import torch 
from core_mlp.module import Parameter,Module
from core_mlp.initialization import xavior_uniform

class Linear(Module):
    def __init__(self,in_features: int, out_features: int, bias: bool = True,device=None,dtype=None):
        super().__init__()      #calling parent class's(Module) constructor(__init__).
        self.in_features = in_features
        self.out_features = out_features
        
        # Only the attributes wrapped as Parametr(..) will be stored in self.parameters_ because-
        # we created __setattr__ function in module class which stored the attributes with -
        # self.parameters_[name] = value 
        # weight and bias are Parameter objects which are now assigned as attributes to self.parameters_

        w = torch.empty(out_features,in_features,device=device,dtype=dtype)
        self.weight = Parameter(xavior_uniform(w))
        if bias:
            b = torch.zeros(out_features,device=device,dtype=dtype)
            self.bias = Parameter(b)
        else:
            self.bias = None
    
    def forward(self,x):
        y = x @ self.weight.T
        if self.bias is not None:
            y = y + self.bias
        return y