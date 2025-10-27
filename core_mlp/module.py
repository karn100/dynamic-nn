from __future__ import annotations
import torch
from typing import Tuple,Dict,Iterator,Any

class Parameter(torch.Tensor):
    #__new__() is used to make 'immutable tensors' subclass to get learnable parameters.
    def __new__(cls,data = None,require_grad = None):
        if data is None:
            data = torch.empty(0)  
        #if data is not a tensor then this convert it to tensor.    
        if not isinstance(data,torch.Tensor):  
            data = torch.as_tensor(data,dtype=torch.float32)
        #This makes a tensor subclass where we made required_grad = True
        params = torch.Tensor._make_subclass(cls,data,require_grad=require_grad)
        return params
    
    #This a developer friendly way to represent the aboe in a well formatted manner.
    def __repr__(self):
        return f"Parameter(shape={tuple(self.shape)}, require_grad={self.require_grad})"

class Module:
    def __init__(self):
        self._parameters: Dict[str,Parameter] = {}
        self._modules: Dict[str,Module] = {}
        self.training: bool = True
    
    def __setattr__(self, name:str, value: Any):
        if isinstance(value,Parameter):
            self._parameters[name] = value
        elif isinstance(value,Module):
            self._modules[name] = value
        super().__setattr__(name,value)
    
    def parameters(self) -> Iterator[Parameter]:
        for param in self._parameters.values():
            yield param
        for module in self._modules.values():
            yield module.parameters()
    
    def to(self,device: torch.device):
        for name,param in self._parameters.items():
            self._parameters[name] = Parameter(param.to(device))
        for module in self._modules.values():
            module.to(device)
        return self
    
    #Training 
    def train(self,mode: bool = True):
        self.training = mode                    #training module instead of parametrs because module(the model) is the container that holds all the parameters, knows how to compute output(forward Pass) and defines how all these parameters work together.
        for module in self._modules.values():
            module.train(mode)
        return self
    
    def eval(self):
        return self.train(False)
    
    def forward(self,*args,**kwargs):

        #This function is empty as this is the main function where we produce the output
        #We define this function when we assign parameters like - return x@self.linear + self.bias
        #If we don't override(define it by re-writnig on it) , then it will throw error below.

        raise NotImplementedError
    
    def __call__(self,*args,**kwargs):
        return self.forward(args,**kwargs)
    
    def __repr__(self):
        mod_str = ','.join(self._modules.keys())
        return f"{__class__.__name__}"(modules = [{mod_str}])
    