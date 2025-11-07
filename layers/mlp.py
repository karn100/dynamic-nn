from core_mlp.module import Module,ModuleList
from layers.linear import Linear
from layers.activation import ReLU

class MLP(Module):
    def __init__(self,in_dims,hidden_dims,out_dims):
        super().__init__()
        dims = [in_dims] + hidden_dims
        layers = []

        for i in range(len(dims) - 1):
            linear = Linear(dims[i],dims[i+1])
            relu = ReLU()
            layers.append(linear)
            layers.append(relu)
        
        out_layer = Linear(dims[-1],out_dims)
        layers.append(out_layer)

        self.layers = ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    

model = MLP(784, [256, 256], 10)

# for name, p in model.named_parameters():
#     print(name, p.shape)