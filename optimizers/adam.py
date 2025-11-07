from __future__ import annotations
import torch
from optimizers.optim import Optimizer

class AdaGrad(Optimizer):
    def __init__(self, params):
        super().__init__(params)