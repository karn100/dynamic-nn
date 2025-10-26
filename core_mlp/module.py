from __future__ import annotations
import torch
from typing import Tuple,Dict,Iterator,Any

class Parameter(torch.Tensor):
    def __new__(cls,data = None,require_grad = None):
        if data is None:
            data = torch.empty(0)  
        if not isinstance(data,torch.Tensor):
            data = torch.as_tensor(data,dtype=torch.float32)
        param = torch.Tensor._make_subclass(cls,data,require_grad=require_grad)
        return param
    
    def __repr__(self):
        return f"Parameter(shape={tuple(self.shape)}, require_grad={self.require_grad})"