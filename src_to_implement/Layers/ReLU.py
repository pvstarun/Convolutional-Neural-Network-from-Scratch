import numpy as np
from Layers.Base import BaseLayer #type:ignore

class ReLU(BaseLayer):
    def __init__(self):
        super().__init__()  # trainable remains False

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        return np.maximum(0, input_tensor)

    def backward(self, error_tensor):
        relu_derivative = self.input_tensor > 0 #stores boolean true values of those that are positive
        return error_tensor * relu_derivative
