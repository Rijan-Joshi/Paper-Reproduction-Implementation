import numpy as np

class Linear:

    def __init__(self, in_features, out_features):

        limit = np.sqrt(6 / (in_features + out_features))
        self.weights = np.random.uniform(-limit, +limit, (in_features, out_features))
        self.bias = np.random.uniform(-limit, +limit, (out_features,))

        self._cache = None
        self.dW = None
        self.db = None
        self.dX = None
    
    def forward(self, X):
        # X: (batch, in_features)
        self._cache = X 
        out = X @ self.weights + self.bias
        return out
        
    def backward(self, dout):
        x = self._cache
        self.dW = dout @ x.T
        self.db = np.sum(dout, axis = 0)
        self.dX = dout @ self.weights.T
        return self.dX