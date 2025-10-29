import torch
from layers.dropout import Dropout

def dropout_test_forward():
    x = torch.ones(10)
    dropout = Dropout(p = 0.5)

    dropout.train()
    y_train = dropout(x)
    assert(y_train.shape == x.shape) # shape remains same because dropping doesnt means it will remove the values from tensor, it will make their values = 0.
    assert(y_train <= 2).all()

    dropout.eval()
    y_eval = dropout(x)
    assert torch.allclose(y_eval,x)  # during eval() , dropout layer becomes inactive and it does not drop random neurons

    print("Test Passed")

if __name__ == "__main__":
    dropout_test_forward()
