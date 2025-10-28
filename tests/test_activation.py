import torch
from layers.activation import ReLU,Sigmoid,Tanh

def activation_test_forward():
    x = torch.tensor([-1.0,1,0,2.0])
    acts = [ReLU(),Sigmoid(),Tanh()]
    for act in acts:
        y = act(x)
        assert isinstance(y,torch.Tensor)
        assert y.shape == x.shape
    print("All Activation test passed")

if __name__ == "__main__":
    activation_test_forward()
    
