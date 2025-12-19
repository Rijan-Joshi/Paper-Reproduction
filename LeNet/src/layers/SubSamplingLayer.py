#Subsampling Layer for the Model

from src.utils.ScaledTanh import ScaledTanh
from src.utils.cupy_numpy import np

class SubSample:
    """
        LeNet-5 Based Sub-sampling layer with trainable parameters and bias
        Takes the no. of channels and pool_size
        Returns the same number of channels after pooling
    """

    def __init__(self, n_channels, pool_size = 2):
    
        self.tanh = ScaledTanh()
        self.n_channels = n_channels
        self.pool_size = pool_size

        self.alpha = np.ones((n_channels,))
        self.beta = np.zeros((n_channels,))

        #Caching for backprop
        self.X = None
        self.pooled = None
        self.Z  = None
    
    def forward(self, X):
        self.X = X
        channels = self.n_channels
        P = self.pool_size
        batch, _ , h, w = X.shape
        
        h_out = h // P
        w_out = w // P

        out = np.zeros((batch, channels, h_out, w_out))

        #More Efficient method (Vectorized)
        for i in range(h_out):
            for j in range(w_out):
                h_start = i * P
                w_start = j * P
                patch = X[:, :, h_start: h_start + P, w_start: w_start + P]
                out[:, :, i, j] = np.mean(patch, axis = (2,3))

        # region
        # for n in range(batch):
        #     for o in range(channels):
        #         for h in range(0, h_out):
        #             for w in range(0, w_out):
        #                 h_start = h * P
        #                 w_start = w * P

        #                 patch = X[n, o, h_start:h_start+P, w_start:w_start+P]
        #                 Z[n, o, h, w] = np.mean(patch)
        # endregion
        self.pooled = out    
        self.Z = np.zeros_like(out)
    
        for channel in range(channels):
            self.Z[:, channel] = self.alpha[channel] * out[:, channel] + self.beta[channel]

        self.A = self.tanh.forward(self.Z)

        return self.A

    def backward(self, dA):
        """
            Backward Pass for SubSampling 
        """

        N, C, H, W = dA.shape

        dZ = self.tanh.backward(dA)
        K = self.pool_size

        d_alpha = np.zeros_like(self.alpha)
        d_beta = np.zeros_like(self.beta)
        H_in = H * K
        W_in = W * K
        dX = np.zeros((N, C, H_in, W_in))

        for i in range(H):
            for j in range(W):

                d_alpha += np.sum(dZ[:, : , i, j] * self.pooled[:, :, i, j], axis = 0)
                d_beta += np.sum(dZ[:, :, i , j], axis = 0)

                for di in range(K):
                    for dj in range(K):
                        start_h = i * K
                        start_w = j * K
                        dX[:, :, start_h + di, start_w+dj] = dZ[:, :, i, j] * self.alpha * (1 / (K * K))
                
        self.grads = {
            'alpha': d_alpha // N,
            'beta': d_beta // N
        }

        return dX

