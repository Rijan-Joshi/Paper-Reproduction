from src.utils.cupy_numpy import np

class ScaledTanh:

    def __init__(self, A = 1.7159, B = (2.0/3.0)):
        self.X = None
        self.A = A
        self.B = B

    def forward(self, X):
        self.X = X
        return self.A * (np.tanh(self.B * X))

    def backward(self, dout):
        return dout * self.A * self.B * (1 - np.tanh(self.B * self.X)**2)