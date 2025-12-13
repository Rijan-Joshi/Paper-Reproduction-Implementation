import numpy as np

class Linear:

    def __init__(self, in_features, out_features):

        limit = np.sqrt(6 / (in_features + out_features))
        self.W = np.random.uniform(-limit, +limit, (in_features, out_features))
        self.b = np.random.uniform(-limit, +limit, (out_features,))

        self._cache = None
        self.dW = None
        self.db = None
        self.dX = None
    
    def forward(self, X):
        # X: (batch, in_features)
        self._cache = X 
        out = X @ self.W + self.b
        print("Shape out of Linear Layer", out.shape)
        return out # Shape: (batch, out_features)
        
    def backward(self, dout):
        print("---" * 10)
        x = self._cache 
        print(dout.shape, x.shape)
        batch = dout.shape[0] 
        self.dW = x.T @ dout #Shape : (in_features, out_features)
        self.db = np.sum(dout, axis = 0)  #Shape: (out_features, )
        self.dX = dout @ self.W.T # Shape: (batch, in_features)

        self.grads = {
            'W' : self.dW / batch, 
            'b' : self.db / batch
        }

        return self.dX