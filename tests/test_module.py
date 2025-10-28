import torch
import os,sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"..")))
from core_mlp.module import Module,Parameter

def test_module_registration():
    class Dummy(Module):
        def __init__(self):
            super().__init__()
            self.w = Parameter(torch.randn(3,3))
            self.b = Parameter(torch.randn(3))
        
        def forward(self, x):
            return x@self.w + self.b
    
    model = Dummy()
    assert(len(list(model.parameters()))) == 2