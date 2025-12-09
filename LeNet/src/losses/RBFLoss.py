import numpy as np

class RBFLoss:

    def __init__(self)
        self._cache = None
    
    def forward(self, rbf_output, y_true, j = 1):
        """
            rbf_output: (N, C)
            y_true: (N,)
        """

        batch = rbf_output[0]
        correct_distances = rbf_output[np.arange(batch), y_true] # shape: (N,)

        junk = np.exp(-j) #scalar
        garbage = np.sum(np.exp(-rbf_output), axis = 1) # shape (N, )
        log_term = np.log(junk + garbage)

        per_sample = correct_distances + log_term
        loss = np.mean(per_sample)

        return loss, per_sample
    
    def backward(self, dout):
        ...

        