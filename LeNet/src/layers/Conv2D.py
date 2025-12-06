import numpy as np


class Conv2D:
    """
    2D Convolutional Layer with nested loops
    im2col is also implemented but not used for the original implementation
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1): ...

    def forward(self): ...

    def backward(self): ...
