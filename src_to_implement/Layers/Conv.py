import numpy as np
import scipy.signal
import copy
from .Base import BaseLayer


class Conv(BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels): #stride_shape = (1, 1), convolution_shape = (3, 2, 2), num_kernels = 2
        super().__init__()
        self.weights = np.random.rand(num_kernels, *convolution_shape) #(2, 3, 2, 2)  # (num_kernels, num_channels, kernel_height, kernel_width)
        self.bias = np.random.random(num_kernels) # (2,)  # one bias per kernel  
        self.total_kernels = num_kernels # 2
        self.stride_shape = stride_shape # (1, 1)
        self.convolution_shape = convolution_shape # (3, 2, 2) 
        self.trainable = True
        self.output_tensor_batch = None
        self._optimizer = None
        self._gradient_weights = 0
        self._gradient_bias = 0

    def forward(self, input_tensor):
        self.input_tensor = input_tensor #(32,3,3,3)
        output_tensor_batch = []
        batch_size = np.shape(input_tensor)[0] #32
        for b in range(batch_size):
            output_tensor_conv = []
            for n_k in range(self.total_kernels): #2
                output = scipy.signal.correlate(input_tensor[b], self.weights[n_k], 'same') #to maintain the same output size as input size # (3, 3, 3)
                output = output[output.shape[0] // 2] + self.bias[n_k] #since we are doing 0 padding, we have to take the middle layer where
                #the kernal and input completely overlap and then add the bias to it # (3, 3, 3).
                if (len(self.stride_shape) == 1):
                    output = output[::self.stride_shape[0]]  # start to end skip 1 row and column #downsampling the output according to stride 
                    output_tensor_conv.append(output)
                elif (len(self.stride_shape) == 2):
                    output = output[::self.stride_shape[0], ::self.stride_shape[1]] # start to end skip 1 row and column #downsampling the output according to stride
                    output_tensor_conv.append(output)
            output_tensor_batch.append(output_tensor_conv)
            self.output_tensor_batch = np.array(output_tensor_batch) # converting to numpy array
        return self.output_tensor_batch #(32, 2, 3, 3)

    @ property          
    def optimizer(self):
        return self._optimizer
    @ optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    def backward(self,error_tensor): # error_tensor = (32, 2, 3, 3)
        batch_size = np.shape(error_tensor)[0] # 32
        num_kernels = self.convolution_shape[0] # 3
        num_channels = self.total_kernels # 2
        weights = np.swapaxes(self.weights, 1, 0) # (3, 2, 2, 2) swapping the first and second axis to prepare for convolution with the error tensor
        weights = np.flip(weights, 1) # (3, 2, 2, 2) flipping the weights along the second axis to prepare for convolution with the error tensor
        updated_error_batch = np.zeros((batch_size, num_channels, *self.input_tensor.shape[2:])) # (32, 2, 3, 3) creating a zero tensor to store the updated error tensor after applying the stride
        output_tensor =[]
        for b in range(batch_size):
            if (len(self.stride_shape) == 1):
                updated_error_batch[:, :, ::self.stride_shape[0]] = error_tensor[b] # upsampling the error tensor according to the stride to prepare for convolution with the weights
            else:
                updated_error_batch[:, :, ::self.stride_shape[0], ::self.stride_shape[1]] = error_tensor[b]
            output_tensor_conv = []
            for n_k in range(num_kernels):
                output = scipy.signal.convolve(updated_error_batch[b], weights[n_k], 'same') # (3, 3, 3) convolving the upsampled error tensor with the flipped weights to get the updated error tensor for the previous layer
                output = output[output.shape[0] // 2] 
                output_tensor_conv.append(output) 
            output_tensor.append(output_tensor_conv)
        self._gradient_weights = self.update_gradient_weights(error_tensor) 
        if self._optimizer is not None:
            self.optimizer_weight = copy.deepcopy(self._optimizer)
            self.optimizer_bias = copy.deepcopy(self._optimizer)
            self.weights = self.optimizer_weight.calculate_update(self.weights, self._gradient_weights)
            self.bias = self.optimizer_bias.calculate_update(self.bias, self._gradient_bias)
        return np.array(output_tensor)

    def update_gradient_weights(self,error_tensor):
        batch_size = np.shape(error_tensor)[0] # 32
        temp_weights = 0
        num_channels = self.total_kernels # 2
        update_error_batch = np.zeros((batch_size, num_channels, *self.input_tensor.shape[2:])) # (32, 2, 3, 3) creating a zero tensor to store the updated error tensor after applying the stride
        for b in range(batch_size):
            if (len(self.stride_shape) == 1):
                update_error_batch[:, :, ::self.stride_shape[0]] = error_tensor[b]
                self._gradient_bias = np.sum(error_tensor, axis=(0, 2))
                pading_X = np.pad(self.input_tensor[b],
                                  ((0, 0), (self.convolution_shape[1] // 2, (self.convolution_shape[1] - 1) // 2)),
                                  'constant', constant_values=0)
            else:
                update_error_batch[:, :, ::self.stride_shape[0], ::self.stride_shape[1]] = error_tensor[b]
                self._gradient_bias = np.sum(error_tensor, axis=(0, 2, 3))
                pading_X = np.pad(self.input_tensor[b],
                                  ((0, 0), (self.convolution_shape[1] // 2, (self.convolution_shape[1] - 1) // 2),
                                   (self.convolution_shape[2] // 2, (self.convolution_shape[2] - 1) // 2)),
                                  'constant', constant_values=0)
            all_gradient_kernels = []
            for n_c in range(num_channels):  
                each_gradient_kernel = []
                for channel_X in range(self.input_tensor.shape[1]):
                    gradient_weight = scipy.signal.correlate(pading_X[channel_X], update_error_batch[b][n_c], 'valid')
                    each_gradient_kernel.append(gradient_weight)
                all_gradient_kernels.append(each_gradient_kernel)
            all_gradient_kernels = np.array(all_gradient_kernels)
            temp_weights += all_gradient_kernels
        return temp_weights


    @property
    def gradient_weights(self):
        return self._gradient_weights
    @ gradient_weights.setter
    def gradient_weights(self, gradient_weights):
        self._gradient_weights = gradient_weights
    @property
    def gradient_bias(self):
        return self._gradient_bias


    def initialize(self, weights_initializer, bias_initializer):
        self.weights = weights_initializer.initialize(self.weights.shape, np.prod(self.convolution_shape),self.total_kernels * np.prod(self.convolution_shape[1:]))
        self.bias = bias_initializer.initialize(self.bias.shape, 1, self.total_kernels)