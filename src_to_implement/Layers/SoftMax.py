import numpy as np
from Layers.Base import BaseLayer #type:ignore

class SoftMax(BaseLayer):
    def __init__(self):
        super().__init__()
        #self.output = None

    def forward(self, input_tensor):
        # For numerical stability: subtract max from each row
        input_stable = input_tensor - np.max(input_tensor, axis=1, keepdims=True)
        exp_tensor = np.exp(input_stable)
        self.output = exp_tensor / np.sum(exp_tensor, axis=1, keepdims=True)
        return self.output

    def backward(self, error_tensor):
        # Jacobian-vector product for softmax and cross entropy
        '''batch_size, num_classes = self.output.shape
        gradient = np.empty_like(error_tensor)

        for i in range(batch_size):
            y = self.output[i].reshape(-1, 1)
            jacobian = np.diagflat(y) - np.dot(y, y.T)
            gradient[i] = np.dot(jacobian, error_tensor[i])'''
        dot = np.sum(self.output * error_tensor,
                 axis=1,
                 keepdims=True)

        return self.output * (error_tensor - dot)
        
        #return gradient
    