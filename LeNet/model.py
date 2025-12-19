#LeNet-5 Collective Architecture
from src.layers.Conv2D import Conv2D
from src.layers.SubSamplingLayer import SubSample
from src.layers.Flatten import Flatten
from src.layers.Linear import Linear
from src.layers.RBFLayer import RBF
from src.losses.RBFLoss import RBFLoss
from src.utils.bitmap_prototypes import get_all_digits_as_array
from src.utils.ScaledTanh import ScaledTanh
from src.utils.cupy_numpy import np

def get_c3_connection_table():
    connection_table =  [
            [0, 1, 2],        # map 0
            [1, 2, 3],        # map 1
            [2, 3, 4],        # map 2
            [3, 4, 5],        # map 3
            [0, 4, 5],        # map 4
            [1, 2, 5],        # map 5
            [0, 1, 3, 4],     # map 6
            [1, 2, 4, 5],     # map 7
            [0, 1, 2, 5],     # map 8
            [0, 2, 3, 5],     # map 9
            [1, 3, 4, 5],     # map 10
            [0, 1, 4, 5],     # map 11
            [0, 1, 2, 3],     # map 12
            [0, 1, 2, 3, 4],  # map 13
            [1, 2, 3, 4, 5],  # map 14
            [0, 1, 2, 3, 4, 5]  # map 15
        ]

    return connection_table


class LeNet5:

    def __init__(self):

        self.connection_table = get_c3_connection_table()
        self.bitmaps = get_all_digits_as_array()

        #Model Architecture
        self.C1 = Conv2D(1, 6, 5)
        self.S2 = SubSample(6, 2)
        self.C3 = Conv2D(6, 16, 5, connection_table=self.connection_table)
        self.S4 = SubSample(16, 2)
        self.C5 = Conv2D(16, 120, 5)
        self.Flatten = Flatten()
        self.Linear = Linear(120, 84)
        self.ScaledTanh = ScaledTanh()
        self.RBF = RBF(84, 10, self.bitmaps)
        self.MAP_Loss = RBFLoss()

        self.layers = [self.C1, self.S2, self.C3, self.S4, self.C5, self.Flatten, self.Linear, self.RBF]

    
    def forward(self, X):
        """
            Forward pass through the network
        """

        X = self.C1.forward(X) # (batch, 6, 28, 28)
        X = self.S2.forward(X)  # (batch, 6, 14, 14)
        X = self.C3.forward(X) # (batch, 16, 10, 10)
        X = self.S4.forward(X)  # (batch, 16, 5, 5)
        X = self.C5.forward(X)   # (batch, 120, 1, 1)
        X = self.Flatten.forward(X) # (batch, 120)
        X = self.Linear.forward(X) #F6 # (batch, 84)
        X = self.ScaledTanh.forward(X) # (batch, 84)
        Y = self.RBF.forward(X) # (batch, 10)

        return Y

    def backward(self, dout):
        """ Backward pass through the network """
        # print("--"*10)
        # print(f"0 {dout.shape}")
        dout = self.RBF.backward(dout)
        # print(f"1 {dout.shape}")
        dout = self.ScaledTanh.backward(dout)
        # print(f"2 {dout.shape}")
        dout = self.Linear.backward(dout)
        # print(f"3 {dout.shape}")
        dout = self.Flatten.backward(dout)
        # print(f"4 {dout.shape}")
        dout = self.C5.backward(dout)
        # print(f"5 {dout.shape}")
        dout = self.S4.backward(dout)
        # print(f"6 {dout.shape}")
        dout = self.C3.backward(dout)
        # print(f"7 {dout.shape}")
        dout = self.S2.backward(dout)
        # print(f"8 {dout.shape}")
        dout = self.C1.backward(dout)
        # print(f"9 {dout.shape}")

        return dout

    def compute_loss(self, Y, y_true):
        loss, _ = self.MAP_Loss.forward(Y, y_true)
        return loss
    
    def predict(self, X):
        Y = self.forward(X)
        predictions = np.argmin(Y, axis = 1)
        return predictions

    def get_accuracy(self, X, y):
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy

    def get_named_params_and_grads(self):
        named_params = []
        grads = []

        #C1-Layer
        named_params += [('C1.W', self.C1.W), ("C1.b", self.C1.b)]
        grads += [self.C1.grads['W'], self.C1.grads['b']]
        
        #S2-Layer
        named_params += [('S2.alpha', self.S2.alpha), ("S2.beta", self.S2.beta)]
        grads += [self.S2.grads['alpha'], self.S2.grads['beta']]

        #C3-Layer
        for i, w in enumerate(self.C3.W):
            named_params += [(f'C3.W{i}', w)]
            grads += [self.C3.grads['W'][i]]
        named_params += [('C3.b', self.C3.b)]
        grads += [self.C3.grads['b']]

        #S4-Layer
        named_params += [('S4.alpha', self.S4.alpha), ("S4.beta", self.S4.beta)]
        grads += [self.S4.grads['alpha'], self.S4.grads['beta']]

        #C5-Layer
        named_params += [('C5.W', self.C5.W), ("C5.b", self.C5.b)]
        grads += [self.C5.grads['W'], self.C5.grads['b']]

        #F6 - Linear
        named_params += [('F6.W', self.Linear.W), ('F6.b', self.Linear.b)]
        grads += [self.Linear.grads['W'], self.Linear.grads['b']]
        
        return named_params, grads

        
        
