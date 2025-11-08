import torch
from core_mlp.module import Parameter
from optimizers.SGD import SGD
from optimizers.momentum import Momentum,MomentumwithNAG
from optimizers.adam import AdaGrad,RMSProp,Adam

w = Parameter(torch.randn(()),require_grad=True)
optimizer = Adam([w],lr=0.1,beta1=0.9,beta2=0.99,eps=1e-8)  #adjusting lr according to params and optimizer is very important

for step in range(100):

    y_pred = w * 2
    loss = (y_pred - 2)**2

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

print("Learned Weight",w.item())