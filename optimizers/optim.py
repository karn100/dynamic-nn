from __future__ import annotations
import torch
from typing import Iterable

class Optimizer:

    # this is the method which is used in real world optimizers where we can assign different lr to diferent layers
    #

    def __init__(self,params,lr):
        if isinstance(params,dict) or hasattr(params,"data"):
            params = [{"params":params}]  # this assign params(weights,bias lr in dict like {"weight":w1,w2,..})
        elif isinstance(params,(list,tuple)): #this se is the params are list or tuple of parameters like -- params = [w1,w2..]
            if not isinstance(params[0],dict): #if tehe first element of list is not dict then 
                params = [{"params":params}]   # make it a dict. if it is a group of param which are list then no need to do this.For eg  -  {"params": [...], "lr": ...}, {"params": [...], "lr": ...}

        #this makes "lr" a default group in params either a user define it or not , it will be there as default
        for group in params:      
            group.setdefault("lr",lr)
        
        self.params_group = params

    def zero_grad(self):
        # This is because, during backward pass- we dont overwrite grads but accumulate them, so if we have previous computed grad,
        # it will be added to the next computed grad which will make training unstable and messy,
        # So, thats why we zero the previous grad and it will not make any impact on the current grad

        for group in self.params_group:
            for p in group["params"]:
                if p.grad is not None:
                    p.grad.zero()
    def step(self):

        #It performs single optimization step
        # This is the function that we overwriite according to which optimizer we are implementing

        raise NotImplementedError
    
    def state_dict(self):
        return {"params_group":self.params_group}
    
    def load_state_dict(self,state):
        self.params_group = state["params_group"]