import numpy as np

class Pooling:
    def __init__(self, stride_shape, pooling_shape):
        self.trainable = False  # Pooling layers have no trainable parameters
        self.stride_shape = self._ensure_tuple(stride_shape) # (1,1)
        self.pooling_shape = self._ensure_tuple(pooling_shape) # (2,2)

        # Stored for backward pass
        self.input_tensor_shape = None
        self.max_indices = None # To store the indices of the maximum values

    def _ensure_tuple(self, shape):
        """Ensures shape is a tuple, converting single int to tuple if needed."""
        if isinstance(shape, int):
            return (shape,)
        return tuple(shape)

    def forward(self, input_tensor):
        # Store the shape of the input tensor for the backward pass
        self.input_tensor_shape = input_tensor.shape # (32,3,3,3)

        batch_size, channels, input_y, input_x = input_tensor.shape
        pool_y, pool_x = self.pooling_shape 
        stride_y, stride_x = self.stride_shape 

        # Calculate output dimensions for "valid" padding
        # output_dim = (input_dim - pool_dim) // stride_dim + 1
        output_y = (input_y - pool_y) // stride_y + 1 #(3-2)//1 + 1 = 2
        output_x = (input_x - pool_x) // stride_x + 1 # (3-2)//1 + 1 = 2

        # Initialize output tensor and max_indices tensor
        output_tensor = np.zeros((batch_size, channels, output_y, output_x)) # (32, 3, 2, 2)
        # max_indices will store the (y, x) coordinates of the max value within each pooling window
        # We need to store these indices for each element in the output_tensor
        self.max_indices = np.zeros((batch_size, channels, output_y, output_x, 2), dtype=int) # Stores (y_idx, x_idx) (32, 3, 2, 2, 2) 2 because of (y, x) coordinates

        for b in range(batch_size): # 32
            for c in range(channels): # 3
                for oy in range(output_y): # 2
                    for ox in range(output_x): # 2
                        # Define the pooling window
                        start_y = oy * stride_y # 0, 1
                        end_y = start_y + pool_y # 2, 3
                        start_x = ox * stride_x # 0, 1
                        end_x = start_x + pool_x # 2, 3

                        # Extract the window
                        window = input_tensor[b, c, start_y:end_y, start_x:end_x] # (2, 2)

                        # Find the maximum value in the window
                        max_value = np.max(window)
                        output_tensor[b, c, oy, ox] = max_value

                        # Find the relative index of the maximum value within the window
                        # np.unravel_index converts a flat index to a coordinate tuple
                        relative_max_idx = np.unravel_index(np.argmax(window), window.shape)

                        # Store the absolute index in the original input_tensor
                        # This is (start_y + relative_y, start_x + relative_x)
                        self.max_indices[b, c, oy, ox, 0] = start_y + relative_max_idx[0] # y-coordinate
                        self.max_indices[b, c, oy, ox, 1] = start_x + relative_max_idx[1] # x-coordinate

        return output_tensor

    def backward(self, error_tensor):
        # Initialize the error tensor for the previous layer with zeros
        # This tensor will have the same shape as the input_tensor from the forward pass
        error_previous_layer = np.zeros(self.input_tensor_shape)

        batch_size, channels, output_y, output_x = error_tensor.shape #(32, 3, 2, 2)

        for b in range(batch_size):
            for c in range(channels):
                for oy in range(output_y):
                    for ox in range(output_x):
                        # Get the error value for this pooling output
                        error_value = error_tensor[b, c, oy, ox]

                        # Get the stored absolute index of the maximum value for this pooling window
                        max_y_idx = self.max_indices[b, c, oy, ox, 0]
                        max_x_idx = self.max_indices[b, c, oy, ox, 1]

                        # Propagate the error value only to the position that held the maximum
                        error_previous_layer[b, c, max_y_idx, max_x_idx] += error_value

        return error_previous_layer
