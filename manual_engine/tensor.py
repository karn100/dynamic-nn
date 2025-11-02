import numpy as np

class Tensor:
    def __init__(self,data,requires_grad = False,parents = (),op = None):
        self.data = data
        self.requires_grad = requires_grad
        self.parents = parents
        self.grad = None
        self.op = op
        self.backward_fn = None
        

        