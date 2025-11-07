import torch
from core_mlp.module import Parameter
from optimizers.SGD import SGD
from optimizers.momentum import Momentum,MomentumwithNAG
from optimizers.adam import AdaGrad,RMSProp

w = Parameter(torch.randn(()),require_grad=True)
optimizer = RMSProp([w],lr=0.1,alpha=0.9,eps=1e-8)

for step in range(100):

    y_pred = w * 1
    loss = (y_pred - 2)**2

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

print("Learned Weight",w.item())