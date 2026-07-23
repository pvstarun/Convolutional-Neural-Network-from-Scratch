import numpy as np
from Layers.Base import BaseLayer # type: ignore

class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True
        self.weights = np.random.rand(input_size + 1, output_size)  # +1 for bias
        self._optimizer = None
        self.input_size = input_size
        self.output_size = output_size
    
    def initialize(self, weights_initializer, bias_initializer):
        # Determine fan_in and fan_out for weights
        fan_in_weights = self.input_size
        fan_out_weights = self.output_size

        # Initialize the main weight matrix (excluding bias)
        # Assuming bias is the last row or column. Let's assume it's the last row for simplicity.
        # If your implementation stores bias as the first row, adjust accordingly.
        weights_shape_without_bias = (self.input_size, self.output_size)
        initialized_weights_core = weights_initializer.initialize(weights_shape_without_bias, fan_in_weights, fan_out_weights)

        # Initialize bias
        bias_shape = (1, self.output_size) # Bias is a row vector for each output neuron
        initialized_bias = bias_initializer.initialize(bias_shape, fan_in_weights, fan_out_weights) # fan_in/out might not be directly applicable for bias, but pass for consistency

        # Combine weights and bias into the self.weights matrix
        # Assuming bias is appended as the last row
        self.weights = np.vstack((initialized_weights_core, initialized_bias)) # vstack combines the weight matrix and bias vector vertically

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        batch_size = input_tensor.shape[0]
        bias = np.ones((batch_size, 1)) # Create a column of ones for the bias term
        input_with_bias = np.concatenate((input_tensor, bias), axis=1) # Adding bias term to the input tensor as a column of ones
        self.input_with_bias = input_with_bias
        return np.dot(input_with_bias, self.weights) # X.W

    def backward(self, error_tensor):
        self._gradient_weights = np.dot(self.input_with_bias.T, error_tensor) #X.T . dL/dY
        if self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.weights, self._gradient_weights)
        return np.dot(error_tensor, self.weights.T[:, :-1]) # dL/dX = dL/dY . W.T (excluding bias weights)

    @property
    def gradient_weights(self):
        return self._gradient_weights
