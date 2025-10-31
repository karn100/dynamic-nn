import torch
from core_mlp.module import Parameter
from optimizers.SGD import SGD
from optimizers.momentum import Momentum,MomentumwithNAG

w = Parameter(torch.randn(()),require_grad=True)
optimizer = MomentumwithNAG([w],lr=0.1,momentum=0.9)

for step in range(100):

    y_pred = w * 1
    loss = (y_pred - 2)**2

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

print("Learned Weight",w.item())