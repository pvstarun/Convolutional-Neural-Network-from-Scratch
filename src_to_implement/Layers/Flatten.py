import numpy as np

class Flatten:
    def __init__(self):
        # Flatten layer is not trainable, so set trainable to False
        self.trainable = False
        # Store the shape of the input tensor during the forward pass
        # This is needed to reshape the error tensor correctly during the backward pass
        self.input_shape = None

    def forward(self, input_tensor):
        # Store the original shape of the input tensor (excluding the batch dimension)
        # The batch dimension (input_tensor.shape[0]) should be preserved
        #(m,3,28,28) self.input_shape = (3, 28, 28)
        self.input_shape = input_tensor.shape[1:]

        # Reshape the input tensor:
        # The first dimension (batch size) remains the same.
        # The remaining dimensions are flattened into a single dimension.
        # np.prod(self.input_shape) calculates the product of all elements in input_shape,
        # effectively giving the total number of features after flattening.
        #reshaping input tensor to (m, 784x3) or (m, 2352)
        output_tensor = input_tensor.reshape(input_tensor.shape[0], -1) # The -1 argument tells numpy to calculate the size of this dimension automatically based on the other dimensions and the total number of elements in the array.
        return output_tensor

    def backward(self, error_tensor):
        # Reshape the error tensor back to the original input shape
        # The batch dimension (error_tensor.shape[0]) remains the same.
        # The remaining flattened dimension is reshaped back to the stored input_shape.
        #backward_tensor (m, 3, 28,28) retains the original shape
        backward_tensor = error_tensor.reshape(error_tensor.shape[0], *self.input_shape)
        return backward_tensor
