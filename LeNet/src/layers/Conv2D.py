import numpy as np

class Conv2D:
    """
    2D Convolutional Layer with nested loops
    im2col is also implemented but not used for the original implementation
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Scaling based on Xavier/ Glorot Initialization
        fan_in = self.in_channels * self.kernel_size * self.kernel_size
        fan_out = self.out_channels * kernel_size * kernel_size

        # This limit is the range from -limit to +limit which weight can use to initialize themselves
        limit = np.sqrt(6 / (fan_in + fan_out))

        # Initializations
        # We need the n sets of k * k weight matrix (n = in_channels)
        self.W = np.random.uniform(
            -limit, +limit, (out_channels, in_channels, kernel_size, kernel_size)
        )
        self.b = np.zeros((out_channels,))

        # Cache - store for backprop
        self.X = None
        self.Z = None

    def forward(self, X):
        """
        Forward pass for Convolutional Layer, which is basically convolution operation

        Batch, Channel, height, width -> X
        Return -> Batch, Out_Channel, height, width
        """

        batch, _, height, width = X.shape
        K = self.kernel_size
        s = self.stride

        out_height = (height - K) // s + 1
        out_width = (width - K) // s + 1

        Z = np.zeros((batch, self.out_channels, out_height, out_width))

        #More Effiencient Method (Vectorized)
        for h in range(out_height):
            for w in range(out_width):
                        height_start = h * s
                        width_start = w * s
                        patch = X[:, :, height_start:height_start+K, width_start:width_start+K]
                        Z[:,:, h, w] = np.einsum('oikl,bikl->bo', self.W, patch) + self.b

        # region
        # Convolution Operation and Extraction of Feature Map
        # for n in range(batch):
        #     for o in range(self.out_channels):
        #         for i in range(out_height):
        #             for j in range(out_width):
        #                 height_start = i * s
        #                 width_start = j * s
        #                 height_end = height_start + K
        #                 width_end = width_start + K

        #                 # Extracting patch from the nth image
        #                 patch = X[n, :, height_start:height_end, width_start:width_end]

        #                 # Create the feature map after convolution operation
        #                 Z[n, o, i, j] = np.sum(patch * self.W[o]) + self.b[o]
        # endregion

        self.Z = Z
        # LeNet-5 used scaled squashing tanh function as below
        self.A = 1.7159 * np.tanh((2 / 3) * Z)
        return self.A

    def backward(self, dA):
        """
        Backward Pass
        Calculation of gradient for back-propagation and optimization
        """

        batch, channels, height, width = dA.shape
        K = self.kernel_size
        S = self.stride

        # Derivative of tanh is 1 - tanh^2
        dZ = dA * 1.7159 * (2 / 3) * (1 - np.tanh((2 / 3) * self.Z) ** 2)

        dW = np.zeros_like(self.W)
        db = np.zeros_like(self.b)
        dX = np.zeros_like(self.X)

        for n in range(batch):
            for o in range(channels):
                for h in range(height):
                    for w in range(width):
                        start_h = h * S
                        start_w = w * S
                        end_h = start_h + K
                        end_w = start_w + K

                        # Get the patch
                        patch = self.X[n, :, start_h:end_h, start_w:end_w]
                        dW[o] += dZ[n, o, h, w] * patch  # Gradient w.r.t weights
                        db[o] += dZ[n, o, h, w]  # Gradient w.r.t bias
                        dX[n, :, start_h:end_h, start_w:end_w] += (
                            dZ(n, o, h, w) * self.W[o]
                        )

        self.grads = {"W": dW / batch, "b": db / batch}
        return dX
