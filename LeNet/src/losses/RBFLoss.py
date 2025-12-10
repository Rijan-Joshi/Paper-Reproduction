import numpy as np

class RBFLoss:

    def __init__(self):
        self._cache = None
 
    
    def forward(self, rbf_output, y_true, j = 1):
        """
            rbf_output: (N, C)
            y_true: (N,)
        """

        batch = rbf_output[0]
        correct_distances = rbf_output[np.arange(batch), y_true] # shape: (N,)

        junk = np.exp(-j) 
        garbage = np.sum(np.exp(-rbf_output), axis = 1) # shape (N, )
        log_term = np.log(junk + garbage)

        per_sample = correct_distances + log_term
        loss = np.mean(per_sample)
    
        self._cache = (rbf_output, y_true)

        return loss, per_sample
    
    def backward(self):
        ...

        rbf_output, y = self._cache

        batch = rbf_output.shape[0]
        j = 1
        junk = np.exp(-j)
        exp_neg = np.exp(-rbf_output)
        garbage = np.sum(exp_neg, axis = 1, keepdims = True) #Shape: (N, 1)

        denom = junk + garbage

        dout = np.zeros_like(rbf_output)

        dout[np.arange(batch), y] = 1

        dout -= exp_neg/garbage

        return dout


        