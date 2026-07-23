import copy

class NeuralNetwork:
    def __init__(self, optimizer, weights_initializer=None, bias_initializer=None):
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
        self.optimizer = optimizer
        self.loss = []
        self.layers = [] #eg [FullyConnected, ReLU, FullyConnected, Softmax] decided on UnitTest file
        self.data_layer = None #stores all the data
        self.loss_layer = None

    def forward(self):
        input_tensor, label_tensor = self.data_layer.next() #gives the next mini batch of data and labels
        output = input_tensor #in the beginning output is simply the input tensor
        for layer in self.layers:
            output = layer.forward(output) # forward pass through each layer, each layer receives the output of the previous layer
        loss = self.loss_layer.forward(output, label_tensor) # calculate the loss using the output of the last layer and the true labels
        self.label_tensor = label_tensor  # Save for backward
        return loss

    def backward(self):
        error_tensor = self.loss_layer.backward(self.label_tensor) # compute the initial error tensor from the loss layer
        for layer in reversed(self.layers): #eg [Softmax, FullyConnected, ReLU, FullyConnected]
            error_tensor = layer.backward(error_tensor) # each layer receives the error tensor from the next layer and computes its own error tensor to pass to the previous layer

    def append_layer(self, layer):
        if layer.trainable:
            layer.optimizer = copy.deepcopy(self.optimizer)
            layer.initialize(self.weights_initializer, self.bias_initializer)
        self.layers.append(layer)

    def train(self, iterations):
        for _ in range(iterations):
            loss = self.forward()
            self.backward()
            self.loss.append(loss)

    def test(self, input_tensor):
        output = input_tensor
        for layer in self.layers:
            output = layer.forward(output)
        return output
