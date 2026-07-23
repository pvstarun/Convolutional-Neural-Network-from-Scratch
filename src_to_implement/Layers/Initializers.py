import numpy as np

class Constant:
    def __init__(self, value=0.1):
        self.value = value

    #Shape = Weights Shape
    def initialize(self, shape, fan_in, fan_out):
        #sets the weight matrix to 0.1 (only good for bias)
        return np.full(shape, self.value) # np.full creates an array of the given shape and fills it with the specified value.

class UniformRandom:
    def initialize(self, shape, fan_in, fan_out):
        return np.random.rand(*shape) # shape is a tuple, so we use *shape to unpack it into the arguments of np.random.rand

class Xavier:
    def initialize(self, shape, fan_in, fan_out):
        # Xavier initialization good for tanh/sigmoid activations
        # Scale or sigma = sqrt(2 / (fan_in + fan_out))
        scale = np.sqrt(2 / (fan_in + fan_out))
        return np.random.normal(0, scale, shape)

class He:
    def initialize(self, shape, fan_in, fan_out):
        # He initialization good for ReLU activations
        # Scale or sigma = sqrt(2 / fan_in)
        scale = np.sqrt(2 / fan_in)
        return np.random.normal(0, scale, shape)
