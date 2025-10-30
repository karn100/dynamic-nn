import torch
from core_mlp.module import Parameter
from optimizers.SGD import SGD

w = Parameter(torch.randn(()),require_grad=True)
optimizer = SGD([w],lr=0.1)

for step in range(100):

    y_pred = w * 3
    loss = (y_pred - 2)**2

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

print("Learned Weight",w.item())