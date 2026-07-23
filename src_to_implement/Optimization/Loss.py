import numpy as np

class CrossEntropyLoss:
    def __init__(self):
        pass

    def forward(self,prediction_tensor, label_tensor):
        x=label_tensor
        y=prediction_tensor+np.finfo(np.float64).eps
        self.y=y
        mask = x > 0 #boolean mask
        #print(mask)
        y = y[mask]  
        cross_entropy = -np.sum(np.log(y))
        return cross_entropy
    
    def backward(self,label_tensor):
        return -(label_tensor/self.y)
    