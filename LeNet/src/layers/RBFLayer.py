from src.utils.bitmap_prototypes import *

class RBF:

    def __init__(self, in_units = 84, num_classes = 10, bitmaps = get_all_digits_as_array()):
        self.bitmaps = bitmaps
        self.in_units = in_units
        self.num_classes = num_classes

        self.W = bitmaps.reshape(bitmaps.shape[0], -1)
        self.X = None
    
    def forward(self, X):
        self.X = X
        N = X.shape[0]
        out = np.zeros((N, self.num_classes))

        #RBF = sum((xi - wij)^2)
        X_norm = np.sum(X**2, axis = 1, keepdims = True) # N, 1
        W_norm = np.sum(self.W**2, axis = 1, keepdims = True) # 10, 1

        out = X_norm + W_norm.T - 2 * X @ self.W.T # N,C

        return out

    def backward(self, dout):
        """
            dout: (N, C)
            dX: (N, 84)
        """
        
        dX_norm = 2 * self.X * np.sum(dout, axis = 1, keepdims = True)
        dX_dot = -2 * (dout @ self.W)

        dX = dX_norm + dX_dot

        return dX