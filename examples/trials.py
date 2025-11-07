import numpy as np
class AdaGrad:
    def __init__(self,params,lr = 0.001,eps = 1e-8):
        self.params = params
        self.lr = lr
        self.eps = eps
        self.G = [np.zeros_like(p) for p in params]

    def step(self,grads):
        for i in range(len(self.params)):
            self.G[i] += grads[i]**2
            adjusted_lr = self.lr / np.sqrt(self.G[i] + self.eps)
            self.params[i] -= adjusted_lr * grads[i]


class RMSProp:
    def __init__(self,params,lr = 0.01,eps = 1e-8,gamma = 0.9):
        self.params = params
        self.lr = lr
        self.eps = eps
        self.gamma = gamma

        self.ema = [np.zeros_like(p) for p in params]
    
    def step(self,grads):
        for i in range(len(self.params)):
            self.ema[i] = self.gamma*self.ema[i] + (1 - self.gamma)*grads[i]**2
            adjusted_lr = self.lr / np.sqrt(self.ema[i] + self.eps)
            self.params[i] -= adjusted_lr * grads[i] 

theta = np.array(10.0)
optimizer = RMSProp(params=[theta],lr=0.01,gamma=0.9)
for step in range(1,21):
    grads = 2 * (theta - 3)
    optimizer.step([grads])
    print(f"Step {step:2d}: theta = {theta:.5f}")

