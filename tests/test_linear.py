import torch
from layers.linear import Linear

def test_linear_forward():
    layer = Linear(3,4)
    x = torch.randn(5,3)
    y = layer(x)
    assert y.shape == (5,4)
    assert isinstance(y,torch.Tensor)
    print("fowrard test passed")

if __name__ == "__main__":
    test_linear_forward()

