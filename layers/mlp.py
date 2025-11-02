from core_mlp.module import Module
from .linear import Linear
from .activation import ReLU

class MLP(Module):
    def __init__(self,in_dims,hidden_dims,out_dims):
        super().__init__()
        layers = []
        #dims = features , in_dims = input features which is fitted in list here.
        #hidden_dims = hidden_features which are genrally in more layers so ,[hid1,hid2]
        
        dims = [in_dims] + hidden_dims  # this helps in addition -- in_dims = [4], hidden_dims = [16,34]
                                        #                           dims = [4,16,34] 

        #for each hidden layer , we add 2 things in sequence -
        #A linear layer and a activation layer    
        for i in range([dims] - 1):
            layers.append(Linear(dims[i],dims[i + 1]))
            layers.append(ReLU())
        
        layers.append(Linear(dims[-1],out_dims))
        layers.append = layers
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
        