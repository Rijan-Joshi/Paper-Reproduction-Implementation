import numpy as np

class SDLM:
    """
        Stochastic Diagonal Levenbeg Marquardt Method Implementation for Optimization of the Network
    """

    def __init__(self, lr = 1.0, mu = 0.01, decay = 0.05, eps = 1e-8):
        self.lr = lr
        self.damping = mu
        self.weight_decay = decay
        self.eps = eps
        self.hessian_diag = {}
        self.lr_multipliers = {}
    
    def set_learning_rate(self, name, lr):
        self.lr_multipliers[name] = lr
    
    def step(self, named_params, grads):
        ...