import numpy as np

class LinearRegression:
    def __init__(self, input_dim = 1):
        self.weights = np.zeros(input_dim)
        self.bias = np.zeros(1)

    def forward(self, X):
        return np.dot(X, self.weights) + self.bias
    
    def get_params(self):
        return np.concatenate((self.weights, self.bias))
    
    def set_params(self, params):
        split_idx = self.weights.size
        self.weights = params[:split_idx].reshape(self.weights.shape)
        self.bias = params[split_idx:].reshape(self.bias.shape)

    def grads(self, X, loss_grad):
        return np.concatenate((np.dot(X.T, loss_grad), [np.sum(loss_grad)]))

