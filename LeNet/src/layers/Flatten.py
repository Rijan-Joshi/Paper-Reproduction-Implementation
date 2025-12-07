
class Flatten:

    def __init__(self,):
        self.shape = shape
    
    def forward(self, X):
        self.shape = X.shape
        return X.reshape(X.shape[0], -1)
    
    def backward(self, dout):
        return dout.reshape(self.shape)

        