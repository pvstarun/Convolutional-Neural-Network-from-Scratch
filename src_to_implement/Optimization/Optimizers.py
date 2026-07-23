import numpy as np

class Sgd:
    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor: np.ndarray, gradient_tensor: np.ndarray) -> np.ndarray:
        return weight_tensor - self.learning_rate * gradient_tensor
    
class SgdWithMomentum:
    def __init__(self, learning_rate, momentum_rate):
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.v = None  # Stores the momentum vector

    def calculate_update(self, weight_tensor, gradient_tensor):
        # Initialize v (velocity) with zeros if it's the first iteration
        if self.v is None:
            self.v = np.zeros_like(weight_tensor)

        # Calculate the new velocity
        # v = momentum_rate * v - learning_rate * gradient_tensor
        self.v = self.momentum_rate * self.v - self.learning_rate * gradient_tensor

        # Update weights
        # W = W + v
        return weight_tensor + self.v

class Adam:
    def __init__(self, learning_rate, mu, rho):
        self.learning_rate = learning_rate
        self.mu = mu      # Often referred to as beta1
        self.rho = rho    # Often referred to as beta2
        self.m = None     # First moment vector (mean)
        self.v = None     # Second moment vector (uncentered variance)
        self.t = 0        # Time step (iteration counter)
        self.epsilon = 1e-8 # Small constant to prevent division by zero

    def calculate_update(self, weight_tensor, gradient_tensor):
        self.t += 1  # Increment time step

        # Initialize m and v with zeros if it's the first iteration
        if self.m is None:
            self.m = np.zeros_like(weight_tensor)
            self.v = np.zeros_like(weight_tensor)

        # Update biased first moment estimate (m)
        # m = mu * m + (1 - mu) * gradient_tensor
        self.m = self.mu * self.m + (1 - self.mu) * gradient_tensor

        # Update biased second moment estimate (v)
        # v = rho * v + (1 - rho) * (gradient_tensor ** 2)
        self.v = self.rho * self.v + (1 - self.rho) * (gradient_tensor ** 2)

        # Correct bias for m
        # m_hat = m / (1 - mu^t)
        m_hat = self.m / (1 - self.mu ** self.t)

        # Correct bias for v
        # v_hat = v / (1 - rho^t)
        v_hat = self.v / (1 - self.rho ** self.t)

        # Update weights
        # W = W - learning_rate * m_hat / (sqrt(v_hat) + epsilon)
        return weight_tensor - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
